import torch
import torch.nn as nn
import sys
sys.path.append("D:/playgroundv25")
from source.model.convolution import ConvBlock, TransposeConvBlock
import random


class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channel_list,
                 out_channel_list,
                 normalization_list,
                 kernel_size_list,
                 stride_list,
                 padding_list):
        super(ConvEncoder, self).__init__()
        self.num_layers = len(in_channel_list)
        self.layers = nn.ModuleList([ConvBlock(in_channel=in_channel_list[index],
                                               out_channel=out_channel_list[index],
                                               normalization=normalization_list[index],
                                               kernel_size=kernel_size_list[index],
                                               stride=stride_list[index],
                                               padding=padding_list[index]) for index in range(self.num_layers)])

    def init_weight(self):
        for layer in self.layers:
            layer.init_weight()

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, in_channel_list, out_channel_list, kernel_size_list, stride_list, padding_list):
        super(ConvDecoder, self).__init__()
        self.num_layers = len(in_channel_list)
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            activation = nn.ReLU() if index < self.num_layers - 1 else nn.Sigmoid()
            layer = TransposeConvBlock(in_channel=in_channel_list[index],
                                       out_channel=out_channel_list[index],
                                       kernel_size=kernel_size_list[index],
                                       stride=stride_list[index],
                                       padding=padding_list[index],
                                       activation=activation)
            self.layers.append(layer)

    def init_weight(self):
        for layer in self.layers:
            layer.init_weight()

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, encoder_in_channel_list, encoder_out_channel_list,
                 encoder_kernel_size_list, encoder_padding_list, encoder_normalization_list,
                 encoder_stride_list, decoder_in_channel_list, decoder_out_channel_list,
                 decoder_stride_list, decoder_kernel_size_list, decoder_padding_list):
        super(ConvAutoencoder, self).__init__()
        self.encoder = ConvEncoder(in_channel_list=encoder_in_channel_list,
                                   out_channel_list=encoder_out_channel_list,
                                   normalization_list=encoder_normalization_list,
                                   kernel_size_list=encoder_kernel_size_list,
                                   stride_list=encoder_stride_list,
                                   padding_list=encoder_padding_list)
        self.decoder = ConvDecoder(in_channel_list=decoder_in_channel_list,
                                   out_channel_list=decoder_out_channel_list,
                                   kernel_size_list=decoder_kernel_size_list,
                                   stride_list=decoder_stride_list,
                                   padding_list=decoder_padding_list)

    @staticmethod
    def from_config(config):
        return ConvAutoencoder(encoder_in_channel_list=config.ENCODER.IN_CHANNEL_LIST,
                               encoder_out_channel_list=config.ENCODER.OUT_CHANNEL_LIST,
                               encoder_kernel_size_list=config.ENCODER.KERNEL_SIZE_LIST,
                               encoder_padding_list=config.ENCODER.PADDING_LIST,
                               encoder_normalization_list=config.ENCODER.NORMALIZATION_LIST,
                               encoder_stride_list=config.ENCODER.STRIDE_LIST,
                               decoder_in_channel_list=config.DECODER.IN_CHANNEL_LIST,
                               decoder_out_channel_list=config.DECODER.OUT_CHANNEL_LIST,
                               decoder_stride_list=config.DECODER.STRIDE_LIST,
                               decoder_kernel_size_list=config.DECODER.KERNEL_SIZE_LIST,
                               decoder_padding_list=config.DECODER.PADDING_LIST)

    def init_weight(self):
        self.encoder.init_weight()
        self.decoder.init_weight()

    def forward(self, _in):
        x = self.encoder(_in)
        _out = self.decoder(x)
        return _out

    def requires_grad(self, require):
        for param in self.parameters():
            param.requires_grad = require


# if __name__ == "__main__":
#     # test
#     from omegaconf import OmegaConf
#     config_file_path = "config.yaml"
#     config = OmegaConf.load(config_file_path)
#     model = ConvAutoencoder.from_config(config)
#     x = torch.zeros(1, 3, 96, 96)
#     y = model(x)
#     from IPython import embed
#     embed()
