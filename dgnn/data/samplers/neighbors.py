import torch
import copy

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from .minibatch import RandomBatchSampler
from ..nodeblocks import NodeBlocks
from ..transforms import row_norm, col_norm

from .minibatch import RandomBatchSampler, StratifiedMiniBatch, DGLBatchSampler

from ...utils.cython.extension.sparse import sample_neighbors
class NeighborSampler(torch.utils.data.DataLoader):
    
    def __init__(self, 
                adj, 
                batch_size, 
                shuffle=False, 
                num_batches=1, 
                num_layers=1,
                num_neighbors=[],
                minibatch='random',
                **kwargs):
        
        self.data = copy.deepcopy(adj)
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors


        self.partition_meta = ''
        if 'part_meta' in kwargs:
            self.partition_meta = kwargs['part_meta']
            kwargs.pop('part_meta')
        
        if 'node_idx' in kwargs:
            node_idx = kwargs['node_idx']
            kwargs.pop('node_idx')


        if minibatch == 'random':
            self.sampler = RandomBatchSampler(adj.size(0), batch_size, shuffle, num_batches)
        elif minibatch == 'stratified':
            self.sampler = StratifiedMiniBatch(adj.size(0), batch_size, shuffle, num_batches, self.partition_meta)
        elif minibatch == 'dglsim':
            self.sampler = DGLBatchSampler(node_idx, batch_size, shuffle, num_batches)
        
        super().__init__(
            self,
            batch_size=1,
            sampler=self.sampler,
            collate_fn=self.__collate__,
            **kwargs,
        )
        
    def __getitem__(self, idx):
        # Gets the next minibatch (from (minibatch) sampler)
        return idx
    
    def __collate__(self, batch_idx):
        # This function is exectued in parallel and create/modify the graphs...

        batch_nodes = batch_idx[0]
        nodeblocks = NodeBlocks(self.num_layers)
        nodeblocks.set_output_nid(batch_nodes)

        for i in range(self.num_layers):
            batch_adj, next_nodes = self.data.sample_adj(batch_nodes, self.num_neighbors[i])
            # batch_adj = batch_adj.to_symmetric()
            batch_adj = row_norm(batch_adj)
            # batch_adj = gcn_norm(batch_adj)

            nodeblocks.add_layers(batch_adj, next_nodes)
            batch_nodes = next_nodes
        
        # return input_nid, nodeblocks and output_nid
        return batch_nodes, nodeblocks, nodeblocks.output_nid


    def update_k(self, k):
        # print(self.sampler)
        self.sampler.update_k(k)