from torch import nn
from MLP import MLP


class DecoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ffd_hidden, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)

        self.enc_dec_attention = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = MLP(dim_model=dim_model, dim_ffd_hidden=dim_ffd_hidden, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, dec, enc):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(query=dec, key=dec, value=dec)[0]

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. compute encoder - decoder attention
        _x = x
        x = self.enc_dec_attention(query=x, key=enc, value=enc)[0]

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, dim_model, dim_ffd_hidden, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim_model, dim_ffd_hidden, num_heads, dropout)
                                     for _ in range(num_layers)])

    def forward(self, trg, enc_src):
        for layer in self.layers:
            trg = layer(trg, enc_src)
        return trg