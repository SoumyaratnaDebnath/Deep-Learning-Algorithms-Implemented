import torch
import torch.nn as nn

architecture_config = [  
    # (kernel_size, num_filters, stride, padding) 
    (7, 64, 2, 3),
    # M = MaxPool
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",    
    # [(kernel_size, num_filters, stride, padding), ... , num_repeats]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # bias is False because we are using batchnorm
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, kernel_size=kernel_size,
                              stride=stride, padding=padding, padding_mode='reflect') 
        # in the original paper, they did not use batchnorm; since it wasn't invented yet
        self.batchnorm = nn.BatchNorm2d(out_channels) 
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        x = self.leakyrelu(self.batchnorm(self.conv(x)))
        return x
    
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs(**kwargs) # pass arguments in runtime

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == tuple:
                # CNN block
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]
                
            elif type(x) == str:
                # Maxpool
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
                
            elif type(x) == list:
                # CNN block with multiple repeats
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers) # unpacks the list and convert it to Sequential
    
    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496), # in the original paper, it's 4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)), # (S, S, 30) where 30 = 7 (classes) + 5 (boxes)
        )
    

def test(S=7, B=2, C=7):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape) # torch.Size([2, 1470]) 1470 = 7 * 7 * 30


if __name__ == "__main__":
    test()