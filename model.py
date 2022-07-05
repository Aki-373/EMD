import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super.__init__()
        #StyleEncoder
        self.conv1_1 = self.conv_layers(1,64,5,1)
        self.conv1_2 = self.conv_layers(64,128,3,2)
        self.conv1_3 = self.conv_layers(128,256,3,2)
        self.conv1_4 = self.conv_layers(256,512,3,2)
        self.conv1_5 = self.conv_layers(512,512,3,2)
        self.conv1_6 = self.conv_layers(512,512,3,2)
        self.conv1_7 = self.conv_layers(512,512,3,2)
        self.conv1_8 = self.conv_layers(512,512,3,2)
        #ContentEncoder
        self.conv2_1 = self.conv_layers(1,64,5,1)
        self.conv2_2 = self.conv_layers(64,128,3,2)
        self.conv2_3 = self.conv_layers(128,256,3,2)
        self.conv2_4 = self.conv_layers(256,512,3,2)
        self.conv2_5 = self.conv_layers(512,512,3,2)
        self.conv2_6 = self.conv_layers(512,512,3,2)
        self.conv2_7 = self.conv_layers(512,512,3,2)
        self.conv2_8 = self.conv_layers(512,512,3,2)
        #Mixer
        self.W = nn.Parameter(torch.Tensor(np.random.normal(size=(512,512,512))))
        #Decoder
        self.deconv1 = self.conv_trans_layers(1024,512,3,2)
        self.deconv2 = self.conv_trans_layers(1024,512,3,2)
        self.deconv3 = self.conv_trans_layers(1024,512,3,2)
        self.deconv4 = self.conv_trans_layers(1024,512,3,2)
        self.deconv5 = self.conv_trans_layers(1024,256,3,2)
        self.deconv6 = self.conv_trans_layers(512,128,3,2)
        self.deconv7 = self.conv_trans_layers(256,64,3,2)
        self.deconv8 = self.conv_trans_layers(128,3,5,1)

    @staticmethod
    def conv_layers(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        net = nn.Sequential(
             nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return net

    @staticmethod
    def conv_trans_layers(in_channels, out_channels, kernel_size, stride, padding=1):
        net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return net

    def forward(self, x):
        #StyleEncoder
        s1 = self.conv1_1(x)
        s2 = self.conv1_2(s1)
        s3 = self.conv1_3(s2)
        s4 = self.conv1_4(s3)
        s5 = self.conv1_5(s4)
        s6 = self.conv1_6(s5)
        s7 = self.conv1_7(s6)
        s8 = self.conv1_8(s7)
        #ContentEncoder
        c1 = self.conv1_1(x)
        c2 = self.conv1_2(c1)
        c3 = self.conv1_3(c2)
        c4 = self.conv1_4(c3)
        c5 = self.conv1_5(c4)
        c6 = self.conv1_6(c5)
        c7 = self.conv1_7(c6)
        c8 = self.conv1_8(c7)
        #Mixer
        F_i = torch.matmul(s8,self.W)
        F_ij = torch.matmul(F_i,c8)
        #Decoder
        out_0 = torch.cat([F_ij,c8], dim=1)
        out_1 = self.deconv1(out_0)
        out_1 = torch.cat([out_1,c7], dim=1)
        out_2 = self.deconv2(out_1)
        out_2 = torch.cat([out_2,c6], dim=1)
        out_3 = self.deconv3(out_2)
        out_3 = torch.cat([out_3,c5], dim=1)
        out_4 = self.deconv4(out_3)
        out_4 = torch.cat([out_4,c4], dim=1)
        out_5 = self.deconv5(out_4)
        out_5 = torch.cat([out_5,c3], dim=1)
        out_6 = self.deconv6(out_5)
        out_6 = torch.cat([out_6,c2], dim=1)
        out_7 = self.deconv7(out_6)
        out_7 = torch.cat([out_7,c1], dim=1)
        out = self.deconv8(out_7)
        out = nn.Sigmoid(out)

        return out

