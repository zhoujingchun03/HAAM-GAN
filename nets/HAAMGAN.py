
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision.models.mobilenetv2 import _make_divisible

class HADA(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(HADA, self).__init__()

        self.conv_1_1 = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1, stride=1,
                                  padding=0,
                                  bias=True)
        self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_3_2 = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.confusion = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                   bias=True)

        self.relu = nn.GELU()

    def forward(self, x):
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        output1 = torch.cat([output_3_1, output_5_1], 1)
        output2 = self.relu(self.conv_3_2(output1))
        output3 = self.relu(self.conv_3_2(output2))
        output4 = output3 + output1
        output5 = self.relu(self.conv_3_2(output4))
        output6 = output5 + output1
        output7 = self.relu(self.conv_3_2(output6))
        output8 = self.relu(self.conv_1_1(output7))
        output = output8 + x

        return output



class MSA(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(MSA, self).__init__()

        self.conv_3_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.conv_7_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=1, padding=3,
                                  bias=True)
        self.conv_1_1 = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.conv_1_2 = nn.Conv2d(in_channels=in_channels * 3, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                                  bias=True)
        self.cam = SCAM(out_channels)
        self.relu = nn.GELU()

    def forward(self, x):
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))
        output_7_1 = self.relu(self.conv_7_1(x))
        output1 = torch.cat([output_3_1, output_5_1, output_7_1], 1)
        output6 = self.relu(self.conv_1_2(output1))
        output2 = self.cam(output6)
        output3 = torch.cat([output2, output6], 1)
        output4 = self.conv_1_1(output3)
        output = output4 + x
        return output




class SCAM(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SCAM, self).__init__()
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



class CAM(nn.Module):
    def __init__(self, gate_channels):
        super(CAM, self).__init__()
        self.gate_channels = gate_channels
        self.conv_3 = nn.Conv2d(in_channels=gate_channels *2, out_channels=gate_channels, kernel_size=3, stride=1,
                                  padding=1,
                                  bias=True)


    def forward(self, x):
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_avg = F.relu(avg_pool, inplace=True)
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_max = F.relu(max_pool, inplace=True)
        scale = torch.cat([channel_avg, channel_max], 1)
        scale = self.conv_3(scale)
        scale = F.sigmoid(scale)
        return x * scale

class MCA(nn.Module):
    def __init__(self, in_size):
        super(MCA, self).__init__()
        layers = [
            SCAM(in_size),
            CAM(in_size)
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        x = self.model(x)
        return x


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.GELU(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorHAAM(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorHAAM, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.Block1 = HADA(32, 32)
        self.down2 = UNetDown(32, 128)
        self.Block2 = HADA(128, 128)
        self.down3 = UNetDown(128, 256)
        self.Block3 = HADA(256, 256)
        self.down4 = UNetDown(256, 256)
        self.Block = MSA(256, 256)

        self.up1 = UNetUp(256, 256)
        self.seb1 = MCA(512)
        self.up2 = UNetUp(512, 128)
        self.seb2 = MCA(256)
        self.up3 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        b1 = self.Block1(d1)
        d2 = self.down2(b1)
        b2 = self.Block2(d2)
        d3 = self.down3(b2)
        b3 = self.Block3(d3)
        d4 = self.down4(b3)
        b = self.Block(d4)
        u1 = self.up1(b, d3)
        s1 = self.seb1(u1)
        u2 = self.up2(s1, d2)
        s2 = self.seb2(u2)
        u3 = self.up3(s2, d1)
        return self.final(u3)


class DiscriminatorHAAM(nn.Module):

    def __init__(self, in_channels=3):
        super(DiscriminatorHAAM, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            #Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)




