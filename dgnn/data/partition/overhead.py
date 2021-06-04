
import os
import copy
import torch
import math
import numpy as np

import torch_geometric.transforms as T

def overhead(dataset, num_parts, overhead=10):

    data = dataset[0]
    dist_dir = os.path.join(dataset.processed_dir, '../partitioned/')

    partitioned_dir = os.path.join(dist_dir, 'overhead-{}-{}'.format(overhead, num_parts))
    print(partitioned_dir)
    if not os.path.exists(partitioned_dir):
        os.mkdir(partitioned_dir)

    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    val_idx = data.val_mask.nonzero(as_tuple=True)[0]
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]

    train_npp = math.ceil(train_idx.shape[0] / num_parts)
    val_npp = math.ceil(val_idx.shape[0] / num_parts)
    test_npp = math.ceil(test_idx.shape[0] / num_parts)

    train_parts = train_idx.split(train_npp)
    val_parts = val_idx.split(val_npp)
    test_parts = test_idx.split(test_npp)

    for i in range(num_parts):

        part_idx = torch.cat((train_parts[i], val_parts[i], test_parts[i]))

        # add neighbors of part_idx to the part_idx up to overhead%

        tmp_adj = data.adj_t[part_idx]
        tmp_row = torch.unique(tmp_adj.storage.row())
        tmp_col = torch.unique(tmp_adj.storage.col())
        tmp_diff = tmp_col[~tmp_col.unsqueeze(1).eq(tmp_row).any(1)]
        num_overhead = int(overhead * part_idx.size(0) /100)

        # TODO: random or more walk?
        #! only pre-overhead training nodes?
        overhead_nodes = tmp_diff[:num_overhead]
        part_idx = torch.cat((part_idx, overhead_nodes))

        part_adj = data.adj_t[part_idx, part_idx]
        part_feats = data.x[part_idx]
        part_labels = data.y[part_idx]
        part_train_mask = data.train_mask[part_idx]
        part_val_mask = data.val_mask[part_idx]
        part_test_mask = data.test_mask[part_idx]

        torch.save(part_adj, partitioned_dir+'/adj_{}.pt'.format(i))
        torch.save((part_feats, part_labels, part_train_mask, part_val_mask, part_test_mask),
                   partitioned_dir+'/fela_{}.pt'.format(i))