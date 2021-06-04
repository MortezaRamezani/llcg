import torch

from ..layers import DistGConv, DHGConv


class DHGCN(torch.nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 activation,
                 ):
        super().__init__()

        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.activation = activation

        self.layers.append(DHGConv(input_size, hidden_size, 0, 0, num_layers))
        for lid in range(1, num_layers-1):
            self.layers.append(DHGConv(hidden_size, hidden_size, 0,  lid, num_layers))
        self.layers.append(DHGConv(hidden_size, output_size, 0, num_layers-1, num_layers))

        self.rank = 0

    def update_rank(self, rank):

        self.rank = rank
        for layer in self.layers:
            layer.rank = rank

    def forward(self, x, adj, use_hist=False, *args, **kwargs):

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj, use_hist)
            if i < self.num_layers - 1:
                h = self.activation(h)

        return h