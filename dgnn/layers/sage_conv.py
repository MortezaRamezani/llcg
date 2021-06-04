import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
import math

class SAGEConv(nn.Module):
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
        self.linear_self = nn.Linear(input_dim, output_dim, bias=True)
        self.linear_neighbors = nn.Linear(input_dim, output_dim, bias=True)

        # use to distinguish from other layers
        self.graph_layer = True

        if 'layer_id' in kwargs:
            self.layer_id = kwargs['layer_id']

    def forward(self, adj, h, *args):

        out_nodes = adj.size(0)
        self_h = self.linear_self(h[:out_nodes])
        neighbor_h = adj.spmm(h)
        h = self_h + self.linear_neighbors(neighbor_h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.layer_id,
            self.input_dim,
            self.output_dim)