import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial =  nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        ### in_channels * 2 because we send the rgb image and its corresponding map i.e. x, y concatenated along the channels
        
        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature
        
        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )
        
        self.model = nn.Sequential(*layers) # unpack the list to build the model

    def forward(self, x, y): # the y should be either fake or real, and its the job of the discriminator to predict
        x = torch.cat([x, y], dim=1) # concatenatre along the dim 1, which is across the channels
        x = self.initial(x)
        return self.model(x)
    

### test the discriminator
def test_disc():
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    model = Discriminator()
    preds = model(x, y)
    print(preds.shape)

if __name__ == "__main__": test_disc() 