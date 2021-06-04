import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class ResidualLayer(nn.Module):

    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super().__init__()

        self.residual = None

        if 'layer_id' in kwargs:
            self.layer_id = kwargs['layer_id']

    def forward(self, h, *args):
        self.residual = h
        return h

    def __repr__(self):
        return self.__class__.__name__ + f"[{self.layer_id}]"