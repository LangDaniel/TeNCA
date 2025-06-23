""" Parts of the U-Net model """
# parts are taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout, inplace=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, padding, dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel_size, padding, dropout)
        #self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, conv_size=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = (DoubleConv(in_channels=n_channels, out_channels=conv_size, kernel_size=1, padding=0))
        self.down1 = (Down(in_channels=conv_size, out_channels=conv_size*2))
        self.down2 = (Down(in_channels=conv_size*2, out_channels=conv_size*4, dropout=0.5))
        self.down3 = (Down(in_channels=conv_size*4, out_channels=conv_size*8, dropout=0.5))
        #self.down4 = (Down(conv_size*8, conv_size*16))
        #self.up1 = (Up(conv_size*16, conv_size*8))
        self.up1 = (Up(in_channels=conv_size*8, out_channels=conv_size*4, dropout=0.5))
        self.up2 = (Up(conv_size*4, conv_size*2))
        self.up3 = (Up(conv_size*2, conv_size))
        self.outc = (OutConv(conv_size, n_classes))

        self.act = nn.Sigmoid()

    def forward(self, x):
        x = x.transpose(1, -1)      # channels first

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        #x = self.up4(x, x1)
        logits = self.outc(x)

        shape = (logits.shape[0], self.n_classes, *logits.shape[2:4], self.n_channels)
        logits = torch.reshape(logits, shape)
        out = self.act(logits)

        return out
