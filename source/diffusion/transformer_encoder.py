from torch import nn
import sys
sys.path.append("D:/playgroundv25")
from source.diffusion.MLP import MLP


class EncoderLayer(nn.Module):
    def __init__(self, dim_model, dim_ffd_hidden, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = MLP(dim_model=dim_model, dim_ffd_hidden=dim_ffd_hidden,
                       activation=nn.ReLU(inplace=True), dropout=dropout)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(query=x, key=x, value=x)[0]

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, dim_model, dim_ffd_hidden, num_heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim_model, dim_ffd_hidden, num_heads, dropout)
                                     for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x