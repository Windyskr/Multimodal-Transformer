import math

import torch
import torch.nn as nn

# Code adapted from the fairseq repo.

def make_positions(tensor, padding_idx, left_pad):
    max_pos = padding_idx + tensor.size(1)
    device = tensor.device
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, torch.arange(padding_idx + 1, max_pos, device=device))
    buf = getattr(make_positions, buf_name)
    if buf.size(0) < max_pos - padding_idx - 1:
        buf = torch.arange(padding_idx + 1, max_pos, device=device)
        setattr(make_positions, buf_name, buf)

    mask = tensor.ne(padding_idx)
    positions = buf[:tensor.size(1)].unsqueeze(0).expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return positions.masked_fill_(~mask, padding_idx)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()   # device --> actual weight; due to nn.DataParallel :-(
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + seq_len
        device = input.device
        if device not in self.weights or max_pos > self.weights[device].size(0):
            self.weights[device] = self.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx,
            ).to(device)

        positions = make_positions(input, self.padding_idx, self.left_pad)

        return (self.weights[device]
                .to(positions.dtype)
                .index_select(0, positions.view(-1))
                .view(bsz, seq_len, -1)
                .detach())

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number