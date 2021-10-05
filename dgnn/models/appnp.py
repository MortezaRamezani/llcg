import torch
import torch.nn as nn

import torch.nn.functional as f

from ..layers import APPNPConv
from ..data import NodeBlocks

class APPNP(nn.Module):
    """
    APPNP model with simple APPNPConv layers at all layers
    """

    def __init__(self,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 num_layers,
                 activation,
                 layer=APPNPConv,
                 dropout=0,
                 input_norm=False,
                 layer_norm=False,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.layer_type = layer
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        self.batch_norm = nn.BatchNorm1d(features_dim, affine=False)
        self.input_fc = nn.Linear(features_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, num_classes)
        
        for i in range(num_layers):
            self.layers.append(layer(hidden_dim, hidden_dim,))



    def forward(self, x, adjs):
        
        h = x
        adj = adjs

        h = self.activation(self.input_fc(self.batch_norm(h)))
        h_0 = h.clone()

        gcn_cnt = 0

        for i, layer in enumerate(self.layers):

            if type(layer) == self.layer_type:
                
                if type(adjs) == NodeBlocks:
                    adj = adjs[gcn_cnt]
                    gcn_cnt += 1
                
                h = layer(adj, h, h_0)

            else:
                h = layer(h)

        h = self.dropout(h)
        h = self.output_fc(h) 

        return h