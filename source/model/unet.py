import torch
import torch.nn as nn
import torch.nn.init as init
import sys
sys.path.append("D:/playgroundv25")


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def init_weight(self):
        init.xavier_uniform_(self.conv.weight)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super(ResBlock, self).__init__()
        self.block1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, activation=activation,
                                kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout)
        self.block2 = ConvBlock(in_channels=out_channels, out_channels=out_channels, activation=activation,
                                kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout)
        self.res_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                  padding=0) if out_channels != in_channels else None

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + (self.res_conv(x) if self.res_conv else x)

    def init_weight(self):
        self.block1.init_weight()
        self.block2.init_weight()
        if self.res_conv:
            init.xavier_uniform_(self.res_conv.weight)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(DownBlock, self).__init__()
        self.block1 = ResBlock(in_channels=in_channels, out_channels=out_channels, activation=nn.SiLU(inplace=True),
                               kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.block2 = ResBlock(in_channels=out_channels, out_channels=out_channels, activation=nn.SiLU(inplace=True),
                               kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        h1 = self.block1(x)
        h2 = self.block2(h1)
        out = self.down(h2)
        return h1, h2, out

    def init_weight(self):
        self.block1.init_weight()
        self.block2.init_weight()


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.block1 = ResBlock(in_channels=2*in_channels, out_channels=in_channels, activation=nn.SiLU(inplace=True),
                               kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.block2 = ResBlock(in_channels=2*in_channels, out_channels=out_channels, activation=nn.SiLU(inplace=True),
                               kernel_size=3, stride=1, padding=1, dropout=dropout)

    def forward(self, x, h2, h1):
        x = self.up(x)
        x = torch.concat([x, h2], dim=1)
        x = self.block1(x)
        x = torch.concat([x, h1], dim=1)
        x = self.block2(x)
        return x

    def init_weight(self):
        self.block1.init_weight()
        self.block2.init_weight()


class MidBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super(MidBlock, self).__init__()
        self.block1 = ResBlock(in_channels=in_channels, out_channels=hidden_channels, activation=nn.SiLU(inplace=True),
                               kernel_size=3, stride=1, padding=1, dropout=dropout)
        self.block2 = ResBlock(in_channels=hidden_channels, out_channels=out_channels, activation=nn.SiLU(inplace=True),
                               kernel_size=3, stride=1, padding=1, dropout=dropout)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

    def init_weight(self):
        self.block1.init_weight()
        self.block2.init_weight()


class UNet(nn.Module):
    def __init__(self, head_block, down_blocks, mid_block, up_blocks, tail_block):
        super(UNet, self).__init__()
        assert len(down_blocks) == len(up_blocks)
        self.head_block = head_block
        self.down_blocks = nn.ModuleList(down_blocks)
        self.mid_block = mid_block
        self.up_blocks = nn.ModuleList(up_blocks)
        self.tail_block = tail_block
        self.num_feature_maps = len(down_blocks)

    def init_weight(self):
        init.xavier_uniform_(self.head_block.weight)
        for down_block in self.down_blocks:
            down_block.init_weight()
        self.mid_block.init_weight()
        for up_block in self.up_blocks:
            up_block.init_weight()
        init.xavier_uniform_(self.tail_block.weight)

    def forward(self, x):
        x = self.head_block(x)

        feature_maps = []
        for down_block in self.down_blocks:
            h1, h2, x = down_block(x)
            feature_maps.append([h1, h2])

        x = self.mid_block(x)

        feature_maps.reverse()
        for index in range(self.num_feature_maps):
            up_block = self.up_blocks[index]
            h1, h2 = feature_maps[index]
            x = up_block(x, h2, h1)

        x = self.tail_block(x)
        return x


def build_unet(device):
    head_block = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1, padding=2)
    down_block1 = DownBlock(in_channels=128, out_channels=256)
    down_block2 = DownBlock(in_channels=256, out_channels=512)
    down_block3 = DownBlock(in_channels=512, out_channels=1024)

    mid_block = MidBlock(in_channels=1024, hidden_channels=1024, out_channels=1024)

    up_block1 = UpBlock(in_channels=1024, out_channels=512)
    up_block2 = UpBlock(in_channels=512, out_channels=256)
    up_block3 = UpBlock(in_channels=256, out_channels=128)

    tail_block = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, stride=1, padding=0)

    down_blocks = [down_block1, down_block2, down_block3]
    up_blocks = [up_block1, up_block2, up_block3]
    unet = UNet(head_block, down_blocks, mid_block, up_blocks, tail_block)
    return unet.to(device)


if __name__ == '__main__':
    device = torch.device("cuda")
    model = build_unet(device)
    x = torch.randn(13, 3, 96, 96).to(device)
    y = model(x)
    from IPython import embed
    embed()