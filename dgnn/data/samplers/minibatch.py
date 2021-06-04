import torch

from .. import partition as P
class RandomBatchSampler(torch.utils.data.Sampler):
    
    def __init__(self, num_nodes, batch_size, shuffle, num_batches=1):
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_parts = self.num_nodes // self.batch_size
        self.num_batches = num_batches
        self.batched_nodes = self.split_nodes()
        # print(len(self.batched_nodes))
        
    def split_nodes(self):
        # Splits nodes into equal size batches, to ensure each node at least happens once and also iter needs prepared list to iterate
        
        nodes_id = torch.randint(self.num_parts, (self.num_nodes,), dtype=torch.long)
        split_ids = [(nodes_id == i).nonzero(as_tuple=False).view(-1) for i in range(self.num_parts)]
        return split_ids
    
    def select_nodes(self):
        # node_id = torch.randperm(self.num_nodes)
        # return node_id[:self.batch_size]
        nodes_id = torch.randint(self.num_nodes, (self.batch_size, ), dtype=torch.long)
        nodes_id, _  = torch.sort(nodes_id)
        return nodes_id
    
    def __iter__(self):
        # Generates next minibatch, do shuffling here if necessary

        # print('In iter', self.num_batches)
        
        if self.num_batches < 1:
            self.batched_nodes = self.split_nodes()
        else:
            self.batched_nodes = []
            for _ in range(self.num_batches):
                self.batched_nodes.append(self.select_nodes())

        # print(self.batched_nodes)
        return iter(self.batched_nodes)
    
    def __len__(self):
        # Number of minibatch generated
        if self.num_batches < 1 :
            return self.num_parts
        return self.num_batches

    def update_k(self, k):
        self.num_batches = k

class StratifiedMiniBatch(torch.utils.data.Sampler):
    def __init__(self, num_nodes, batch_size, shuffle, num_batches, part_meta, *args):
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_parts = self.num_nodes // self.batch_size
        self.num_batches = num_batches
        
        # load the permutation data
        self.partition_idx = P.load_meta(part_meta)
        self.batched_nodes = None

    def split_nodes(self):
        sampled_nodes = []
        num_partitions = len(self.partition_idx)
        nodes_per_part = self.batch_size // num_partitions
        for i in range(num_partitions):
            perm = torch.randperm(self.partition_idx[i].size(0))
            samples = self.partition_idx[i][perm[:nodes_per_part]]
            sampled_nodes.append(samples)

        batch_nodes, _ = torch.sort(torch.cat(sampled_nodes))
        return batch_nodes

    def __iter__(self):
        if self.num_batches < 1:
            self.batched_nodes = [self.split_nodes()]
        else:
            self.batched_nodes = []
            for _ in range(self.num_batches):
                self.batched_nodes.append(self.split_nodes())

        return iter(self.batched_nodes)

    def __len__(self):
        return len(self.batched_nodes)


class DGLBatchSampler(torch.utils.data.Sampler):
    
    def __init__(self, nodes_idx, batch_size, shuffle, num_batches=1):
        self.node_idx = nodes_idx
        self.num_nodes = len(nodes_idx)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_parts = self.num_nodes // self.batch_size
        self.num_batches = num_batches
        self.batched_nodes = []
    
    def select_nodes(self):
        nodes_id = torch.randint(self.num_nodes, (self.batch_size, ), dtype=torch.long)
        nodes_id, _  = torch.sort(self.node_idx[nodes_id])
        return nodes_id
    
    def __iter__(self):        
        self.batched_nodes = []
        for _ in range(self.num_batches):
            self.batched_nodes.append(self.select_nodes())

        # print(self.batched_nodes)
        return iter(self.batched_nodes)
    
    def __len__(self):
        # Number of minibatch generated
        if self.num_batches < 1 :
            return self.num_parts
        return self.num_batches