import math

from torch import nn
import numpy as np
import torch

from model.args import ARGS
from model.mask_cnn import MaskConv2d

class CrossTransformer(nn.Module):
    def __init__(self, dim, dropout=0.3, use_tri_bias=True, scale=False):
        super().__init__()
        self.h_dim = dim
        self.use_tri_bias = use_tri_bias

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

        # horizontal
        h_mask = mask.view(-1, max_len).sum(dim=-1) != max_len
        tmp_mask = (mask.view(-1, 1, max_len)[h_mask]).bool()  # bsz' x 1 x max_len
        h_attn_mask = mask.view(-1, 1, max_len)[h_mask].float() * -10000.0  # bsz x 1 x max_len x max_len
        h_scores = x.reshape(-1, max_len, dim)
        _h_scores = h_scores[h_mask]  # bsz' x max_len x dim
        __h_scores = self.h_qkv(_h_scores).clamp(-10000, 10000)
        h_q, h_k, h_v = __h_scores.chunk(3, dim=-1)  # bsz' x max_len x dim
        # bsz' x max_len x hsz
        h_attn = torch.matmul(h_q, h_k.transpose(-1, -2)) / self.scale  # bsz' x max_len x max_len
        h_attn = h_attn.clamp(-10000, 10000) + h_attn_mask  # bsz' x max_len x max_len

        # vertical
        t_mask = mask.transpose(1, 2).contiguous()
        v_mask = t_mask.reshape(-1, max_len).sum(dim=-1) != max_len
        v_attn_mask = t_mask.reshape(-1, 1, max_len)[v_mask].float() * -10000.0
        v_scores = x.transpose(1, 2).reshape(-1, max_len, dim)
        _v_scores = v_scores[v_mask]
        _v_scores = self.v_qkv(_v_scores).clamp(-10000, 10000)
        v_q, v_k, v_v = _v_scores.chunk(3, dim=-1)
        v_attn = torch.matmul(v_q, v_k.transpose(-1, -2)) / self.scale  # bsz' x max_len x max_len
        v_attn = v_attn.clamp(-10000, 10000) + v_attn_mask  # bsz' x max_len x max_len

        h_attn = self.dropout(torch.softmax(h_attn, dim=-1)).masked_fill(tmp_mask, 0)
        v_attn = self.dropout(torch.softmax(v_attn, dim=-1)).masked_fill(tmp_mask, 0)
        h_v = torch.matmul(h_attn, h_v)  # bsz' x max_len x max_len,  bsz' x max_len x dim->bsz' x max_len x dim
        v_v = torch.matmul(v_attn, v_v)
        v = torch.cat([h_v, v_v], dim=-1)
        v = self.dense(v)

        v = self.dropout(v)
        _x = torch.zeros_like(x).reshape(-1, max_len, dim)
        _x[v_mask] = v.to(_x)
        v = self.LayerNorm(_x + x.reshape(-1, max_len, dim))

        v = v.reshape(bsz, max_len, max_len, dim).permute(0, 3, 1, 2)
        c_v = self.conv1(v, mask[:, None])
        c_v = self.act(c_v)
        c_v = self.conv2(c_v, mask[:, None]).permute(0, 2, 3, 1).reshape(-1, max_len, dim)
        c_v = self.dropout(c_v) + v.permute(0, 2, 3, 1).reshape(-1, max_len, dim)

        v = self.LayerNorm2(c_v)

        return (v.reshape(bsz, max_len, max_len, dim), mask)
