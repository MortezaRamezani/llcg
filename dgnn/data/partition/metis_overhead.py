import os
import copy
import torch
import math
import numpy as np

def metis_overhead(dataset, num_parts, overhead=10):

    data = dataset[0]

    dist_dir = os.path.join(dataset.processed_dir, '../partitioned/')
    metis_meta_path = os.path.join(dist_dir, f'metis-{num_parts}/perm.pt')

    metis_perm, metis_partptr = torch.load(metis_meta_path)

    full_adj = data.adj_t
    start = 0
    for end in range(num_parts):
        import pdb; pdb.set_trace()
        
        part_idx = metis_perm[start:end]
        tmp_adj = full_adj[part_idx]
        tmp_row = torch.unique(tmp_adj.storage.row())
        tmp_col = torch.unique(tmp_adj.storage.col())
        tmp_diff = tmp_col[~tmp_col.unsqueeze(1).eq(tmp_row).any(1)]
        num_overhead = int(overhead * part_idx.size(0) /100)

        overhead_nodes = tmp_diff[:num_overhead]
        new_part_idx = torch.cat((part_idx, overhead_nodes))


        part_adj = full_adj[part_idx, part_idx]
        part_feats = data.x[part_idx]
        part_labels = data.y[part_idx]
        part_train_mask = data.train_mask[part_idx]
        part_val_mask = data.val_mask[part_idx]
        part_test_mask = data.test_mask[part_idx]        



        start = end

