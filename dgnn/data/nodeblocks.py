import torch
from typing import List, Optional, Tuple
from torch_sparse import SparseTensor

class NodeBlocks():
    """ NodeBlocks hold the adjacency matrix per layer for GCN
    propagation
    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self, num_layers, from_graph=None):
        self.num_layers = num_layers
        self.layers_adj: List[SparseTensor] = []
        self.layers_nodes = []
        self.output_nid = None
        self.is_subgraph = False

        if from_graph is not None:
            self.from_graph(from_graph)

    def __getitem__(self, layer_id):
        return self.layers_adj[layer_id]

    def __len__(self):
        return self.num_layers

    def __repr__(self):
        print(self.layers_adj)
        return ''

    def to(self, device='cpu'):
        """ Move nodeblock to the device
        In case of fullgraph, only first layers is moved and 
        the rest are pointing to the same adjacency
        Keyword Arguments:
            device {str} -- [description] (default: {'cpu'})
        """
        if not self.is_subgraph:
            for i, adj in enumerate(self.layers_adj):
                self.layers_adj[i] = adj.to(device, non_blocking=True)
        else:
            for i, adj in enumerate(self.layers_adj):
                if i == 0:
                    self.layers_adj[i] = adj.to(device, non_blocking=True)
                else:
                    self.layers_adj[i] = self.layers_adj[0]

    def from_graph(self, graph):
        """ Create nodeblocks from a complete graph
        The full adjacency is repeated for all layers.
        It doesn't copy on CPU, but on GPU does.
        To avoid the overhead .to function handle the repeat.
        Arguments:
            graph {[type]} -- [description]
        """

        self.layers_adj = [graph] * self.num_layers
        self.layers_nodes = [torch.arange(graph.size(0))] * self.num_layers
        self.is_subgraph = True

    def add_layers(self, spmx, nodes):
        """ append new layers adjaceny to the nodeblocks
        Arguments:
            spmx {[type]} -- [description]
            nodes {[type]} -- [description]
        """

        self.layers_adj.insert(0, spmx)
        self.layers_nodes.insert(0, nodes)

    def set_output_nid(self, nodes):
        """ set output nid
        Arguments:
            nodes {[type]} -- [description]
        """
        self.output_nid = nodes

    @property
    def input_nid(self):
        return self.layers_nodes[-1]