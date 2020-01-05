import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride, find_mask=False):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

        if find_mask:
            self.conv.weight.data[:, :, :, :] = 1
            self.conv.bias.data[:] = 0


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class WDiscriminator(nn.Module):
    def __init__(self, opt, find_mask=False):
        super(WDiscriminator, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = int(opt.nfc)
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1, find_mask=find_mask)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1, find_mask=find_mask)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)

        if find_mask:
            self.tail.weight.data[:, :, :, :] = 1
            self.tail.bias.data[:] = 0

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
    def __init__(self, opt):
        super(GeneratorConcatSkip2CleanAdd, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        N = opt.nfc
        self.head = ConvBlock(opt.nc_im,N,opt.ker_size,opt.padd_size,1) #GenConvTransBlock(opt.nc_z,N,opt.ker_size,opt.padd_size,opt.stride)
        self.body = nn.Sequential()
        for i in range(opt.num_layer-2):
            N = int(opt.nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N,opt.min_nfc),max(N,opt.min_nfc),opt.ker_size,opt.padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(
            nn.Conv2d(max(N,opt.min_nfc),opt.nc_im,kernel_size=opt.ker_size,stride =1,padding=opt.padd_size),
            nn.Tanh()
        )
    def forward(self,x,y):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind),ind:(y.shape[3]-ind)]
        return x+y


def get_mask_discriminator(real, mask, opt, cover_ratio=0.8):

    real_zeros_ones = copy.deepcopy(real)
    real_zeros_ones[:, :, :, :] = 0
    real_zeros_ones[:, :, mask['xmin']:mask['xmax'] + 1, mask['ymin']:mask['ymax'] + 1] = 1
    toy_disc = WDiscriminator(opt, find_mask=True).to(opt.device)
    output_clamped = torch.clamp(toy_disc(real_zeros_ones), 0, 1)

    xs, ys = [], []
    for i in range(output_clamped.shape[2]):
        for j in range(output_clamped.shape[3]):
            if output_clamped[0, 0, i, j] >= cover_ratio:
                xs.append(i)
                ys.append(j)
    new_mask = {
        'xmin': min(xs),
        'xmax': max(xs),
        'ymin': min(ys),
        'ymax': max(ys)
    }
    return new_mask, output_clamped.shape
