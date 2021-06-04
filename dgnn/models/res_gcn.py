
import torch
import torch.nn as nn

import torch.nn.functional as f

from ..layers import GConv
from ..data import NodeBlocks

from . import GCN


class ResGCN(GCN):
    """
    Residual GCN model
    """

    def __init__(self,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 num_layers,
                 activation,
                 layer=GConv,
                 dropout=0,
                 input_norm=False,
                 layer_norm=False,
                 *args,
                 **kwargs):

        super().__init__(features_dim,
                         hidden_dim,
                         num_classes,
                         num_layers,
                         activation,
                         layer,
                         dropout,
                         input_norm,
                         layer_norm,
                         *args,
                         **kwargs)

    def forward(self, x, adjs):

        h = x
        adj = adjs
        gcl_cnt = 0
        h_res = None

        for i, layer in enumerate(self.layers):

            if hasattr(layer, 'graph_layer'):

                if gcl_cnt > 0:  # TODO: don't clone if it's last gconv layer!
                    h_res = h.clone()

                if isinstance(adjs, NodeBlocks):
                    adj = adjs[gcl_cnt]

                h = layer(adj, h)
                gcl_cnt += 1

            else:
                h = layer(h)
                if i < len(self.layers) - 1 and gcl_cnt > 1 and isinstance(layer, type(self.activation)):
                    # print('ResAdded')
                    h = h + h_res

        return h
