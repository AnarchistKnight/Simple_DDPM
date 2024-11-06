import torch.nn as nn
import torch.nn.init as init


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, normalization, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)
        self.bn = nn.BatchNorm2d(out_channel) if normalization else None
        self.activation = nn.ReLU(inplace=True)
        if in_channel == out_channel and stride == 1 and kernel_size == 2 * padding + 1:
            self.residual = None
        else:
            self.residual = nn.Conv2d(in_channels=in_channel,
                                      out_channels=out_channel,
                                      kernel_size=stride,
                                      stride=stride,
                                      padding=0)

    def init_weight(self):
        init.xavier_uniform_(self.conv.weight)
        if self.residual:
            init.xavier_uniform_(self.residual.weight)

    def shortcut(self, x):
        return self.residual(x) if self.residual else x

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.activation(x)
        x = x + residual
        return x


class TransposeConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, activation):
        super(TransposeConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.TransposeConv = nn.ConvTranspose2d(in_channels=in_channel,
                                                out_channels=out_channel,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding)
        self.activation = activation

    def init_weight(self):
        init.xavier_uniform_(self.TransposeConv.weight)

    def forward(self, x):
        x = self.TransposeConv(x)
        x = self.activation(x)
        return x