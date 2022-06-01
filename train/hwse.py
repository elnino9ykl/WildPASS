import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
Example:(pe means positional encoding)
img = torch.randn(2,64,320,480)
hse = HSE(64, 320, 480, use_pe=True)
wse = WSE(64, 320, 480, use_pe=True)
'''

class HSE(nn.Module):
    def __init__(self, inplanes, H, W, r_h=2, r_c=2, use_pe=False):
        super().__init__()
        self.c = inplanes // r_c
        self.h = H // r_h
        self.w = W
        self.use_pe = use_pe
        self.pool = nn.AvgPool2d(kernel_size=(1,W))
        self.down = nn.AvgPool2d(kernel_size=(r_h, 1))
        self.layer1 = nn.Sequential(nn.Conv2d(inplanes, self.c, kernel_size=1),
                                 nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(self.c, self.c, kernel_size=1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.c, inplanes, kernel_size=1),
                                 nn.Sigmoid())
        self.up = nn.UpsamplingBilinear2d(size=(H, 1))
        self.build_pe()

    def forward(self, x):
        a = self.pool(x)
        a = self.down(a)
        a = self.layer1(a)
        if self.use_pe:
            a += self.pe
        a = self.layer2(a)
        a = self.up(a)
        x = x * a
        return x

    def build_pe(self):
        pe = torch.zeros(self.c, self.h)
        position = torch.arange(0, self.h, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, self.c, step=2, dtype=torch.float) * -(math.log(100.0) / self.c)).unsqueeze(1)
        tmp = div_term * position
        pe[0::2, :] = torch.sin(tmp)
        pe[1::2, :] = torch.cos(tmp) if self.c % 2 == 0 else torch.cos(tmp[:-1])
        pe.unsqueeze_(0).unsqueeze_(3)
        self.register_buffer('pe', pe)

class WSE(nn.Module):
    def __init__(self, inplanes, H, W, r_w=2, r_c=2, use_pe=False):
        super().__init__()
        self.c = inplanes // r_c
        self.h = H
        self.w = W // r_w
        self.use_pe = use_pe
        self.pool = nn.AvgPool2d(kernel_size=(H,1))
        self.down = nn.AvgPool2d(kernel_size=(1, r_w))
        self.layer1 = nn.Sequential(nn.Conv2d(inplanes, self.c, kernel_size=1),
                                 nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(self.c, self.c, kernel_size=1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.c, inplanes, kernel_size=1),
                                 nn.Sigmoid())
        self.up = nn.UpsamplingBilinear2d(size=(1, W))
        self.build_pe()

    def forward(self, x):
        a = self.pool(x)
        a = self.down(a)
        a = self.layer1(a)
        if self.use_pe:
            a += self.pe
        a = self.layer2(a)
        a = self.up(a)
        x = x * a
        return x

    def build_pe(self):
        pe = torch.zeros(self.c, self.w)
        position = torch.arange(0, self.w, dtype=torch.float).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, self.c, step=2, dtype=torch.float) * -(math.log(100.0) / self.c)).unsqueeze(1)
        tmp = div_term * position
        pe[0::2, :] = torch.sin(tmp)
        pe[1::2, :] = torch.cos(tmp) if self.c % 2 == 0 else torch.cos(tmp[:-1])
        pe.unsqueeze_(0).unsqueeze_(2)
        self.register_buffer('pe', pe)
        
        
# inplanes: input channels
class HWSE(nn.Module):
    def __init__(self, inplanes, H, W, r_h=2, r_w=2, r_c=2, use_pe=False):
        super(HWSE, self).__init__()
        self.hse = HSE(inplanes, H, W, r_h, r_c=r_c, use_pe=use_pe)
        self.wse = WSE(inplanes, H, W, r_w, r_c=r_c, use_pe=use_pe)

    def forward(self, x):
        x_h = self.hse(x)
        x_w = self.wse(x)
        return x_h + x_w

