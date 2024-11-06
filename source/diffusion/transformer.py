"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""
import torch
from torch import nn
from transformer_encoder import TransformerEncoder
from transformer_decoder import TransformerDecoder


class Transformer(nn.Module):

    def __init__(self, encoder_num_layers, decoder_num_layers, dim_model, dim_ffd_hidden, num_heads, dropout):
        super().__init__()
        self.encoder = TransformerEncoder(encoder_num_layers, dim_model, dim_ffd_hidden, num_heads, dropout)
        self.decoder = TransformerDecoder(decoder_num_layers, dim_model, dim_ffd_hidden, num_heads, dropout)

    def forward(self, src, trg):
        enc_src = self.encoder(src)
        output = self.decoder(trg, enc_src)
        return output