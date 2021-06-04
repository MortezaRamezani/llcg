import torch
import copy

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from ..nodeblocks import NodeBlocks
from .minibatch import RandomBatchSampler, StratifiedMiniBatch
    
class SubGraphSampler(torch.utils.data.DataLoader):
    
    def __init__(self, 
                adj, 
                batch_size, 
                shuffle=False, 
                num_batches=1, 
                num_layers=1,
                minibatch='random',
                **kwargs):
        
        self.data = copy.deepcopy(adj)
        self.num_layers = num_layers
        
        if 'part_meta' in kwargs:
            self.partition_meta = kwargs['part_meta']
            kwargs.pop('part_meta')

        if minibatch == 'random':
            batch_sampler = RandomBatchSampler(adj.size(0), batch_size, shuffle, num_batches)
        elif minibatch == 'stratified':
            batch_sampler = StratifiedMiniBatch(adj.size(0), batch_size, shuffle, num_batches, self.partition_meta)
        
        super().__init__(
            self,
            batch_size=1,
            sampler=batch_sampler,
            collate_fn=self.__collate__,
            **kwargs,
        )
        
    def __getitem__(self, idx):
        # Gets the next minibatch (from (minibatch) sampler)
        return idx
    
    def __collate__(self, batch_idx):
        # This function is executed in parallel and create/modify the graphs...
        # batch_nodes, _ = torch.sort(batch_idx[0])
        batch_nodes = batch_idx[0]
        
        batch_adj, _ = self.data.saint_subgraph(batch_nodes)
        # batch_adj = batch_adj.to_symmetric()
        batch_adj = gcn_norm(batch_adj.set_value(None))

        batch_nb = NodeBlocks(self.num_layers, batch_adj)
        batch_nb.set_output_nid(batch_nodes)
        
        # return input_nid, nodeblocks and output_nid
        return batch_nodes, batch_nb, batch_nodes