import os
import copy
import torch
import math
import numpy as np

import torch_geometric.transforms as T

def random(dataset, num_parts):
    
    data = dataset[0]
    dist_dir = os.path.join(dataset.processed_dir, '../partitioned/')

    partitioned_dir = os.path.join(dist_dir, 'random-{}'.format(num_parts))
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

    part_meta = []

    for i in range(num_parts):

        part_idx = torch.cat((train_parts[i], val_parts[i], test_parts[i]))
        part_meta.append(part_idx)
        
        # Only saving the diagonal! Fix later!
        part_adj = data.adj_t[part_idx, part_idx]
        part_feats = data.x[part_idx]
        part_labels = data.y[part_idx]
        part_train_mask = data.train_mask[part_idx]
        part_val_mask = data.val_mask[part_idx]
        part_test_mask = data.test_mask[part_idx]

        torch.save(part_adj, partitioned_dir+'/adj_{}.pt'.format(i))
        torch.save((part_feats, part_labels, part_train_mask, part_val_mask, part_test_mask),
                   partitioned_dir+'/fela_{}.pt'.format(i))

    torch.save(part_meta, partitioned_dir+'/perm.pt')