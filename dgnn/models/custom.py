
import torch
import torch.nn as nn

import torch.nn.functional as f

from ..layers import GConv, MLPLayer, SAGEConv #,and more

from ..data import NodeBlocks

class Custom(nn.Module):
    """Custom GNN model builder

    Arguments:
        nn {[type]} -- [description]
    """

    def __init__(self,
                 features_dim,
                 hidden_dim,
                 num_classes,
                 num_layers,
                 activation,
                 layer=None,
                 dropout=0,
                 input_norm=False,
                 layer_norm=False,
                 arch='',
                 *args,
                 **kwargs):

        super().__init__()

        self.num_layers = num_layers
        self.arch = arch
        self.layers = nn.ModuleList()
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        self.residual = kwargs['residual']

        self.model_builder(arch, features_dim, hidden_dim, num_classes, input_norm, layer_norm)

    def model_builder(self, arch, feat_dim, hid_dim, output_dim, input_norm, layer_norm):

        layers_tokens = list(arch)
        num_layers = len(layers_tokens)

        in_dim = feat_dim
        out_dim = hid_dim

        if input_norm:
            self.layers.append(nn.BatchNorm1d(feat_dim, affine=False))
        
        for i, l in enumerate(layers_tokens):            
            
            if l == 'l':
                layer = nn.Linear(in_dim, out_dim)
            elif l == 'g':
                layer = GConv(in_dim, out_dim, layer_id=i)
            elif l == 's':
                layer = SAGEConv(in_dim, out_dim, layer_id=i)

            self.layers.append(layer)
            
            if i < num_layers - 1:
                if layer_norm:
                    self.layers.append(torch.nn.BatchNorm1d(out_dim))
                self.layers.append(self.activation)
                self.layers.append(self.dropout)
            
            in_dim = hid_dim
            out_dim = hid_dim if i + 1 < num_layers - 1 else output_dim


    def forward(self, x, adjs):

        h = x
        adj = adjs
        gcn_cnt = 0
        h_res = None

        for i, layer in enumerate(self.layers):

            if hasattr(layer, 'graph_layer'):
                
                if gcn_cnt > 0 and self.residual:
                    h_res = h.clone()

                if type(adjs) == NodeBlocks:
                    adj = adjs[gcn_cnt]
                
                h = layer(adj, h)
                gcn_cnt += 1

            else:
                h = layer(h)

                if self.residual and i < len(self.layers) - 1 and gcn_cnt > 1 and isinstance(layer, type(self.activation)):
                    h = h + h_res
            
        return h
