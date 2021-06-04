import torch

from ..layers import DistGConv


class DistGCN(torch.nn.Module):

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

        self.layers.append(DistGConv(input_size, hidden_size))
        for _ in range(1, num_layers-1):
            self.layers.append(DistGConv(hidden_size, hidden_size))
        self.layers.append(DistGConv(hidden_size, output_size))

        self.rank = 0

    def update_rank(self, rank):

        self.rank = rank
        for layer in self.layers:
            layer.rank = rank

    def forward(self, x, adj, *args, **kwargs):

        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < self.num_layers - 1:
                h = self.activation(h)

        return h