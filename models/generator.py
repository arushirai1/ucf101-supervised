import torch.nn as nn
import torch
from einops import repeat, rearrange

'''
class Block(nn.Module):
    def __init__(self, in_features, out_features, kernel_sizes, strides, paddings=[0], activation_functions=[], batch_norm_flag=True):
        super(Block, self).__init__()
        paddings = [paddings[0] for i in in_features] if len(paddings) != len(in_features) else paddings
        activation_functions = [nn.ReLU(inplace=True) for i in in_features] if len(activation_functions) != len(in_features) else activation_functions
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(nn.Conv3d(in_feat, out_feat, kernel_size=kernel_size,
                       stride=stride, padding=padding), activation_func)
            for in_feat, out_feat, kernel_size, stride, padding, activation_func in zip(in_features, out_features, kernel_sizes, strides, paddings, activation_functions)])
        if batch_norm_flag:
            self.batch_norm = nn.BatchNorm3d(num_features=out_features[-1], momentum=0.9)
        else:
            self.batch_norm = None

    def forward(self, x):
        x = self.conv_layers(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x
'''

class DecodeBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, midplane=None, dropout=0.3, padding=0, clips=1, batch_norm=True, norm_constant=None):
        super(DecodeBlock, self).__init__()
        if batch_norm:
            self.block = nn.Sequential(nn.ConvTranspose3d(in_features, out_features, kernel_size, stride=stride, padding=padding),
                                       nn.BatchNorm3d(out_features),
                                       nn.ReLU(),
                                       nn.Dropout(dropout))
        else:
            if midplane:
                self.block = nn.Sequential(
                    nn.ConvTranspose3d(in_features, midplane, kernel_size, stride=stride, padding=padding),
                    nn.Conv3d(midplane, out_features, kernel_size=1, stride=1))
            else:
                self.block = nn.Sequential(
                    nn.ConvTranspose3d(in_features, out_features, kernel_size, stride=stride, padding=padding))
        self.norm_constant = norm_constant
        self.clips = clips
    def forward(self, x):
        if len(x.shape) == 3:
            x = rearrange(x, 'b clips f -> b f (clips 1) 1 1')
            x = self.block(x)
        elif len(x.shape) == 5:
            x = self.block(x)
            x = x/self.norm_constant # keep values between 0 and 1
            x = rearrange(x, 'b channels (clips no_frames) h w  -> b clips channels no_frames h w', clips = self.clips)

        return x

class DecodeBlockwithUpsample(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, dropout=0.3):
        super(DecodeBlock, self).__init__()
        self.block = nn.Sequential(nn.Upsample(),
                                   nn.ConvTranspose3d(in_features, out_features, kernel_size, stride=stride),
                                   nn.BatchNorm3d(in_features),
                                   nn.ReLU(),
                                   nn.Dropout(dropout))
    def forward(self, x):

        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, up_sample_size, kernel_size=(3,3,3), stride=1, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.upsample = nn.Upsample(up_sample_size, mode='nearest')
        self.upsample_conv = self._get_conv(in_features, out_features, kernel_size=1, padding=0)
        self.convA = self._get_conv(in_features, out_features,kernel_size, stride)
        self.convB = self._get_conv(out_features, out_features,kernel_size, stride)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    def _get_conv(self, in_features, out_features, kernel_size=(3,3,3), stride=1, padding=1):
        return nn.Sequential(nn.Conv3d(in_features, out_features, kernel_size, stride=stride, padding=padding),
                             nn.BatchNorm3d(out_features))
    def forward(self, x):
        x = self.upsample(x)
        residual = x
        out = self.convA(x)
        out = self.convB(out)
        if self.upsample:
            residual = self.upsample_conv(residual)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out

class MemEfficientConv3D(nn.Module):
    def __init__(self, in_feature, out_feature, kernel_size, stride, padding):
        super(MemEfficientConv3D, self).__init__()
        layers = []
        if out_feature < in_feature:
            layers.append(nn.Conv3d(in_feature, out_feature, kernel_size=1)) # reduce number of feature maps first
            in_feature = out_feature
        layers.append(nn.Conv3d(in_feature, out_feature, kernel_size=kernel_size, stride=stride, padding=padding))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class CNNDecoder(nn.Module):
    def __init__(self, norm_constant):
        super(CNNDecoder, self).__init__()
        self.layers = self._build_layers()
        self.norm_constant = norm_constant

    def _build_block(self, in_feature, midplane, out_feature, kernel, stride, relu=True, momentum=0.9, padding=0):
        if type(kernel) == list:
            layers = [
                MemEfficientConv3D(in_feature, midplane, kernel_size=kernel[0], stride=stride, padding=padding),
                MemEfficientConv3D(midplane, out_feature, kernel_size=kernel[1], stride=stride, padding=padding)
            ]
        else:
            layers = [
                MemEfficientConv3D(in_feature, midplane, kernel_size=kernel, stride=stride, padding=padding),
                MemEfficientConv3D(midplane, out_feature, kernel_size=kernel, stride=stride, padding=padding)
            ]
        if relu:
            layers.insert(1, nn.ReLU())
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm3d(out_feature, momentum=momentum))

        return nn.Sequential(*layers)

    def _build_layers(self):
        layers = []
        layers.append(nn.Upsample((8, 28, 28)))
        layers.append(self._build_block(in_feature=512, midplane=256, out_feature=256, kernel=(5,5,5), stride=(1,1,1), padding=(2,2,2)))
        layers.append(nn.Upsample(scale_factor=(1,2,2)))
        layers.append(self._build_block(in_feature=256, midplane=256, out_feature=128, kernel=(5,5,5), stride=(1,1,1), padding=(2,2,2)))
        layers.append(nn.Upsample(scale_factor=(1,2,2)))
        layers.append(self._build_block(in_feature=128, midplane=64, out_feature=64, kernel=[(5,5,5), (3,3,3)], stride=(1,1,1), padding=(1,1,1)))
        layers.append(nn.Conv3d(64, 3, kernel_size=(1,1,1), stride=(1,1,1), padding=(1,1,1)))
        layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    def forward(self, x, num_frames=8):
        x = torch.repeat_interleave(x, num_frames, dim=1)
        x = rearrange(x, 'b num_frames features -> b features num_frames 1 1')
        x = self.layers(x)
        x = x / self.norm_constant
        x = rearrange(x, 'b channels num_frames h w -> b 1 channels num_frames h w')
        return x

def get_decoder(clips=1, crop_size=112, simple=False, args=None):
    #perhaps we need to resize the input or we resize at the end it will be slower but... it seems like the only correct way to do it
    # how will upsampling work?? not sure... lets keep it limited to 1 clip... scale factor...
    if args.cnndecoder:
        return CNNDecoder(norm_constant=args.normalize_constant)
    layers = []
    if not simple:
        #stem = DecodeBlock(64, 256*3, (3, 8, 8), midplane=64, stride=(1,2,2), padding=(1,3,3), clips=clips, batch_norm=False) # why did the kernel size need to increase
        stem = DecodeBlock(64, 3, (3, 8, 8), stride=(1,2,2), padding=(1,3,3), clips=clips, batch_norm=False, norm_constant=args.normalize_constant) # why did the kernel size need to increase
        if args and args.sigmoid_activation:
            layers = [nn.Linear(512,512),
                             nn.ReLU(),
                             nn.Dropout(0.3),
                             DecodeBlock(512, 512, kernel_size=(1, 7, 7), stride=1), # undo effect of avg pool
                             ResidualBlock(512, 256, up_sample_size=(2,14,14)),
                             ResidualBlock(256, 128, up_sample_size=(4, 28, 28)),
                             ResidualBlock(128, 64, up_sample_size=(8, 56, 56)),
                             stem,
                             nn.Sigmoid()]
        else:
            layers = [nn.Linear(512,512),
                             nn.ReLU(),
                             nn.Dropout(0.3),
                             DecodeBlock(512, 512, kernel_size=(1, 7, 7), stride=1), # undo effect of avg pool
                             ResidualBlock(512, 256, up_sample_size=(2,14,14)),
                             ResidualBlock(256, 128, up_sample_size=(4, 28, 28)),
                             ResidualBlock(128, 64, up_sample_size=(8, 56, 56)),
                             stem]
        return nn.Sequential(*layers)
    else:
        stem = DecodeBlock(512,3, kernel_size=(8,12,12), stride=17, padding=(0,1,1), clips=clips, batch_norm=False) # why did the kernel size need to increase
        return nn.Sequential(nn.Linear(512,512),
                             nn.ReLU(),
                             nn.Dropout(0.3),
                             DecodeBlock(512, 512, kernel_size=(1, 7, 7), stride=1),
                             stem)

'''print("test")
y_hat =decoder1(x)
print(y_hat.shape)

y = torch.rand(5, 3, 8, 112, 112)
import torch.nn.functional as F

y_hat = rearrange(y_hat, 'b clips (channels range) no_frames h w -> b  channels (clips no_frames) h w range', clips=1, channels=3)

print(F.cross_entropy(y_hat, y))
'''
