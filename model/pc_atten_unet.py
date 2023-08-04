import math
import torch
from torch import nn
from torch.nn import functional as F


# 自注意力
class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(in_ch)
        self.proj_q = nn.Conv1d(in_ch, in_ch, 1)
        self.proj_k = nn.Conv1d(in_ch, in_ch, 1)
        self.proj_v = nn.Conv1d(in_ch, in_ch, 1)
        self.proj = nn.Conv1d(in_ch, in_ch, 1)

    def forward(self, x):
        B, C, N = x.shape
        h = self.batch_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 1).view(B, N, C)
        k = k.view(B, C, N)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, N, N]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 1).view(B, N, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, N, C]
        h = h.view(B, N, C).permute(0, 2, 1)
        h = self.proj(h)
        # print(h.shape)
        # print(x.shape)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, attn=False):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 1)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, 1)
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1)
        # if attn:
        #     self.attn = AttnBlock(out_ch)
        # else:
        #     self.attn = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h) + self.shortcut(x)
        # h = self.attn(h)
        return h


class Down(nn.Module):
    def __init__(self, in_ch, md_ch, out_ch, dropout, s_dim, attn=False):
        super().__init__()
        self.res1 = ResBlock(in_ch, md_ch, dropout, attn)
        self.res2 = ResBlock(md_ch, out_ch, dropout, attn)
        self.down_sample = nn.Conv1d(s_dim, s_dim//2, 1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        d = self.down_sample(x.transpose(1, 2))
        return x, d.transpose(1, 2)


class Up(nn.Module):
    def __init__(self, in_ch, md_ch, out_ch, dropout, s_dim, attn=False):
        super().__init__()
        self.res1 = ResBlock(in_ch, md_ch, dropout, attn)
        self.res2 = ResBlock(md_ch, out_ch, dropout, attn)
        self.up_sample = nn.Conv1d(s_dim//8, s_dim, 1)

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        u = self.up_sample(x.transpose(1, 2))
        return u.transpose(1, 2)


class UNet(nn.Module):
    def __init__(self, input_num_pc, output_num_pc, dropout):
        super().__init__()
        self.head = nn.Conv1d(3, 64, 1)
        self.down1 = Down(64, 256, 512, dropout, input_num_pc)

        self.up1 = Up(512, 512, 512, dropout, output_num_pc)
        self.attn = AttnBlock(512)

        self.tail_1 = nn.Sequential(

            nn.Conv1d(512 + 512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 512, 1),
        )

        self.tail_2 = nn.Sequential(

            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 3, 1),

        )

    def forward(self, x):
        x = x.transpose(1, 2)
        h = self.head(x)  # [B, 64, N]
        dx1, d1 = self.down1(h)  # [B, 256, 128]

        f = torch.max(dx1, dim=2, keepdim=True)[0]  # (B, 256, 1)
        # print(f.size())
        u1 = self.up1(d1)  # [B, 1024, 2048]

        # u1 = self.attn(u1)

        # print(u1.size())
        # print(torch.cat([f.expand(-1, -1, 2048), u1], dim=1).size())

        y = self.tail_1(torch.cat([f.expand(-1, -1, 2048), u1], dim=1))  # [B, 256+1024, 2048]

        y = self.attn(y)

        y = self.tail_2(y)

        return y.transpose(1, 2)  # [B, N, 3]


if __name__ == '__main__':
    import time
    # print(time.localtime())

    batch_size = 16
    model = UNet(input_num_pc=512, output_num_pc=2048, dropout=0.1)
    x = torch.randn(batch_size, 512, 3)
    print(time.localtime())

    y = model(x)
    print(time.localtime())

    print(y.shape)
