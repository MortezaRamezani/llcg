
import torch
import torch.nn as nn

import torch.nn.functional as f

from ..layers import GConv
from ..data import NodeBlocks

class GCN(nn.Module):
    """
    GCN model with simple GConv layers at all layers
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

        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        dropout = nn.Dropout(p=dropout)
        self.activation = activation

        self.layer_type = layer


        if input_norm:
            self.layers.append(nn.BatchNorm1d(features_dim, affine=False))


        self.layers.append(layer(features_dim, hidden_dim, layer_id=1))
        if layer_norm:
                self.layers.append(torch.nn.BatchNorm1d(hidden_dim))
        self.layers.append(activation)
        self.layers.append(dropout)

        for i in range(1, num_layers-1):
            self.layers.append(layer(hidden_dim, hidden_dim, layer_id=i+1))
            if layer_norm:
                self.layers.append(torch.nn.BatchNorm1d(hidden_dim))
            self.layers.append(activation)
            self.layers.append(dropout)
        
        self.layers.append(
            layer(hidden_dim, num_classes, layer_id=num_layers))

        # for layer in self.layers:
        #     layer.linear.weight.data.fill_(0.01)

    def forward(self, x, adjs):
        
        h = x
        adj = adjs
        gcn_cnt = 0

        for i, layer in enumerate(self.layers):

            if type(layer) == self.layer_type:
                
                if type(adjs) == NodeBlocks:
                    adj = adjs[gcn_cnt]
                    gcn_cnt += 1
                
                h = layer(adj, h)

            else:
                h = layer(h)
            
        return h