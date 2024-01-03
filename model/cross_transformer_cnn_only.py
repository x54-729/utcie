import math

from torch import nn
import numpy as np
import torch

from model.args import ARGS
from model.mask_cnn import MaskConv2d


class CrossTransformer(nn.Module):
    def __init__(self, dim, dropout=0.3, use_tri_bias=True, scale=False):
        """
        CNN-IE
        """
        super().__init__()
        self.h_dim = dim
        self.use_tri_bias = use_tri_bias
        if use_tri_bias is True:
            pos = torch.ones(512, 512, dtype=torch.long)
            pos.triu_()
            self.register_buffer('pos', pos.long())
            self.pos_embed = nn.Embedding(2, dim)
            nn.init.xavier_normal_(self.pos_embed.weight.data, gain=0.1 if scale else 1)
        if use_tri_bias is 2:
            pos = torch.ones(512, 512, dtype=torch.long) * 2
            pos.triu_()
            pos = pos - torch.eye(512)
            self.register_buffer('pos', pos.long())
            self.pos_embed = nn.Embedding(3, dim)
            nn.init.xavier_normal_(self.pos_embed.weight.data, gain=0.1 if scale else 1)

        self.h_qkv = nn.Linear(dim, 3 * dim)
        self.v_qkv = nn.Linear(dim, 3 * dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(dim)

        self.dense = nn.Linear(2 * dim, dim, bias=True)
        self.LayerNorm = nn.LayerNorm(dim)

        self.conv1 = MaskConv2d(dim, dim, kernel_size=3, padding=1)
        if ARGS.get('use_gelu', False):
            self.act = nn.GELU()
        else:
            self.act = nn.LeakyReLU()
        self.conv2 = MaskConv2d(dim, dim, kernel_size=3, padding=1)
        # self.ffn = nn.Sequential(
        #     nn.Linear(dim, dim*4),
        #     nn.Dropout(0.3),
        #     nn.LeakyReLU(),
        #     nn.Linear(dim*4, dim)
        # )

        self.LayerNorm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """

        :param x: (bsz x max_len x max_len x dim, bsz x max_len x max_len)
        :return:
        """
        x, mask = x
        x = x.clamp(-1000, 1000)
        bsz, max_len, max_len, dim = x.size()

        if self.use_tri_bias:
            pos = self.pos_embed(self.pos[:max_len, :max_len])[None]  # 1 x max_len x max_len x h
            x = pos + x

        v = x.permute(0, 3, 1, 2)
        c_v = self.conv1(v, mask[:, None])
        c_v = self.act(c_v)
        c_v = self.conv2(c_v, mask[:, None]).permute(0, 2, 3, 1).reshape(-1, max_len, dim)
        c_v = self.dropout(c_v) + v.permute(0, 2, 3, 1).reshape(-1, max_len, dim)

        v = self.LayerNorm2(c_v)

        return (v.reshape(bsz, max_len, max_len, dim), mask)

    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        # 如果是旋转query key的话，下面这个直接cat就行，因为要进行矩阵乘法，最终会在这个维度求和。（只要保持query和key的最后一个dim的每一个位置对应上就可以）
        # torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        # 如果是旋转value的话，下面这个stack后再flatten才可以，因为训练好的模型最后一个dim是两两之间交替的。
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)
