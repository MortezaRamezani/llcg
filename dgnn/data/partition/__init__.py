import torch

from .metis import metis
from .random import random
from .overhead import overhead


def load_partitions(parted_path, rank):

    adj = torch.load(parted_path+'/adj_{}.pt'.format(rank))
    features, labels, train_mask, val_mask, test_mask = torch.load(
        parted_path+'/fela_{}.pt'.format(rank))
    return adj, features, labels, train_mask, val_mask, test_mask

def load_meta(parted_path):
    perm = torch.load(parted_path+'/perm.pt')
    if type(perm) == tuple:
        perm, partptr = perm
        all_perm = []
        start = 0
        for end in partptr[1:]:
            all_perm.append(perm[start:end])
            start=end
        return all_perm
    else:
        return perm


# def load_fixed_part(self, rank):
    
#     if not self.parted:
#         # do it once!
#         self.num_nodes = self.dataset[0].num_nodes
#         self.node_per_part = math.ceil(self.num_nodes / self.config.num_procs)
#         self.part_nodes = torch.split(torch.arange(self.num_nodes), self.node_per_part)

#         self.part_ptr = [self.part_nodes[0][0]]
#         for part in self.part_nodes:
#             self.part_ptr.append(part[-1])

#         self.parted = True

#     start = self.part_ptr[rank]
#     if start > 0:
#         start += 1
#     end = self.part_ptr[rank+1]
#     adj = self.dataset[0].adj_t.narrow(0, start, end-start+1).narrow(1, start, end-start+1)


#     part_feats = self.dataset[0].x[start:end+1]
#     part_labels = self.dataset[0].y[start:end+1]
#     part_train_mask = self.dataset[0].train_mask[start:end+1]
#     part_val_mask = self.dataset[0].val_mask[start:end+1]
#     part_test_mask = self.dataset[0].test_mask[start:end+1]

#     return adj, part_feats, part_labels, part_train_mask, part_val_mask, part_test_mask


# def load_random_part(self, rank):

#     if not self.parted:
#         self.parted = True

#         train_idx = self.dataset[0].train_mask.nonzero(as_tuple=True)[0]
#         val_idx = self.dataset[0].val_mask.nonzero(as_tuple=True)[0]
#         test_idx = self.dataset[0].test_mask.nonzero(as_tuple=True)[0]

#         train_npp = math.ceil(train_idx.shape[0] / self.config.num_procs)
#         val_npp = math.ceil(val_idx.shape[0] / self.config.num_procs)
#         test_npp = math.ceil(test_idx.shape[0] / self.config.num_procs)

#         train_parts = train_idx.split(train_npp)
#         val_parts = val_idx.split(val_npp)
#         test_parts = test_idx.split(test_npp)


#         self.part_idx = []
#         for i in range(self.config.num_procs):
#             self.part_idx.append(torch.cat((train_parts[i], val_parts[i], test_parts[i])))

#     part_idx = self.part_idx[rank]
#     part_adj = self.dataset[0].adj_t[part_idx, part_idx]
#     part_feats = self.dataset[0].x[part_idx]
#     part_labels = self.dataset[0].y[part_idx]
#     part_train_mask = self.dataset[0].train_mask[part_idx]
#     part_val_mask = self.dataset[0].val_mask[part_idx]
#     part_test_mask = self.dataset[0].test_mask[part_idx]

#     return part_adj, part_feats, part_labels, part_train_mask, part_val_mask, part_test_mask