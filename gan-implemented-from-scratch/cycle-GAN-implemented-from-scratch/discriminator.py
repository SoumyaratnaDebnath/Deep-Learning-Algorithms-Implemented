import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect", bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.innitial = nn.Sequential( # initial bloack with no instance norm
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = [] # setting up the other layers
        for f_idx in range(1, len(features)):
            layers.append(
                Block(features[f_idx-1], features[f_idx], stride=1 if f_idx == len(features)-1 else 2) 
            )
        
        # in the end the discriminator will either say real of fake, so adding the last layer
        layers.append(
            nn.Conv2d(features[f_idx], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"),
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.innitial(x)
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
    
def test():
    x = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    preds = model(x)
    print(model)
    print(preds.shape) # torch.Size([1, 1, 30, 30])
                       # the authors claims that each value in this 30 by 30 sees a 70 by 70 patch


if __name__=="__main__":
    test() 
