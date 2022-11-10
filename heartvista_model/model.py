import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),  padding=(1,1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out) #Todo: do we want a relu?
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, conv1_out_chan, out_channels, stride=1):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels = conv1_out_chan, kernel_size=(3,3), stride=2, padding=(1,1))
        self.bn1 = nn.BatchNorm2d(conv1_out_chan)
        
        self.pool = nn.MaxPool2d(2) 
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3),  padding=(1,1))
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = self.pool(x)
        out = torch.cat((out1, out2), dim=1)
        
        out = F.relu(self.bn2(self.conv2(out)))
        
        return out

class RegistrationNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        #Pre Concat (ran twice in parallelish)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.resnet1 = ResidualBlock(32, 32)
        self.downsample1 = DownsampleBlock(32, 32, 64)
        self.resnet21 = ResidualBlock(64, 64)
        self.resnet22 = ResidualBlock(64, 64)
        self.downsample2 = DownsampleBlock(64, 64, 128)
        self.resnet3 = ResidualBlock(128, 128)
        
    
        #Post Concat
        self.conv2 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = (3,3),  padding=(1,1))
        self.bn2 = nn.BatchNorm2d(128)
        self.resnet41 = ResidualBlock(128, 128)
        self.resnet42 = ResidualBlock(128, 128)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = (3,3),  padding=(1,1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size = (3,3),  padding=(1,1))
        self.bn4 = nn.BatchNorm2d(2)  
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
    
    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = self.downsample1(self.resnet1(x1))
        x1 = self.downsample2(self.resnet22(self.resnet21(x1)))
        x1 = self.resnet3(x1)
        
        x2 = F.relu(self.bn1(self.conv1(x2)))
        x2 = self.downsample1(self.resnet1(x2))
        x2 = self.downsample2(self.resnet22(self.resnet21(x2)))
        x2 = self.resnet3(x2) 
        
        x = torch.cat((x1, x2), dim=1)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resnet42(self.resnet41(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = 2* F.tanh(self.conv4(x))
                
        return x

