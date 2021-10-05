import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm, SparseTensor
from torch_geometric.utils import softmax
import math


class GATConv(nn.Module):
    """[summary]

    Arguments:
        nn {[type]} -- [description]
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=True)

        # use to distinguish from other layers
        self.graph_layer = True

        self.layer_id = kwargs['layer_id'] if 'layer_id' in kwargs else '0'
        self.num_heads = kwargs['num_heads'] if 'num_heads' in kwargs else 1

        self.attn_src = nn.ModuleList()
        self.attn_dst = nn.ModuleList()

        for _ in range(self.num_heads):
            self.attn_src.append(nn.Linear(self.output_dim, 1, bias=False))
            self.attn_dst.append(nn.Linear(self.output_dim, 1, bias=False))

        self.negative_slope = kwargs['negative_slope'] if 'negative_slope' in kwargs else 0.2


    def forward(self, adj, h, *args):

        h = self.linear(h)
        h_list = []
        
        row = adj.storage.row()
        col = adj.storage.col()
        sparse_sizes = adj.sizes()

        for head in range(self.num_heads):
            attn = self.attn_src[head](h)[row] + self.attn_dst[head](h)[col]
            attn = F.leaky_relu(attn, negative_slope=self.negative_slope)
            attn = softmax(attn, row).flatten()
            attn = SparseTensor(row=row, col=col, value=attn, sparse_sizes=sparse_sizes)
            h_list.append(attn.spmm(h))

        h = torch.cat(h_list, dim=1)
        return h

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.layer_id,
            self.input_dim,
            self.output_dim)