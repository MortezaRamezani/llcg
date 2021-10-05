import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
import math

        
class APPNPConv(nn.Module):
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
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.2

    def forward(self, adj, h, h0, *args):
        output = (1 - self.alpha) * adj.spmm(h)  + self.alpha * h0[:adj.size(0)]
        return output

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.layer_id,
            self.input_dim,
            self.output_dim)