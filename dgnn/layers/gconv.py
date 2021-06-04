import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spmm
import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
class GConv(nn.Module):
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

        if 'layer_id' in kwargs:
            self.layer_id = kwargs['layer_id']

        # self.linear.bias.data.fill_(0)
        # glorot(self.linear.weight)

    def forward(self, adj, h, *args):

        h = adj.spmm(h)
        h = self.linear(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.layer_id,
            self.input_dim,
            self.output_dim)