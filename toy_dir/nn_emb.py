#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-11-28
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
import torch
from torch import nn

emb = nn.Embedding(10, 3)

ans = emb(torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.long))
print(type(ans.shape[0]))  # 4 2 3
print(ans.ndimension())  # 4 2 3
print(torch.arange(3))


def _n_false(n, m):
    '''https://www.thinbug.com/q/55330169'''
    return np.array([False] * n + [True] * (m - n))


n_false = np.vectorize(_n_false, signature='(),()->(n)')
ans = n_false(
    np.array([0, 1, 2, 3]),
    3
)
print(ans)


class OrdinalEmbedding(nn.Embedding):
    def forward(self, input: torch.Tensor):
        assert input.ndimension() == 2
        assert input.shape[1] == 1
        N = input.shape[0]
        K = self.num_embeddings
        M = self.embedding_dim
        # [N, K, M]
        all_emb = super(OrdinalEmbedding, self).forward(torch.arange(K, dtype=torch.long)[None, :].repeat(N, 1))
        all_emb=all_emb.abs_()
        mask = torch.from_numpy(np.tile(n_false(input.detach().numpy().flatten(), K)[:, :, np.newaxis], [1, 1, M]))
        all_emb[mask] = 0
        out = torch.sum(all_emb, dim=1)
        if self.training:
            mean = torch.mean(out, dim=0)
            if getattr(self, 'init', True):
                self.running_mean = mean
            else:
                self.init = False
                self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * mean
            # get output
        else:
            mean = self.running_mean
        return out - mean


ord_emb_layer = OrdinalEmbedding(10, 2)
ans = ord_emb_layer(torch.tensor([[0], [1], [2], [3]], dtype=torch.long))
ans=ans.detach().numpy()
import pylab as plt

for i in range(ans.shape[0]):
    plt.scatter(ans[i,0],ans[i,1],label=f'{i}')
print(ans)
plt.legend()
plt.show()
