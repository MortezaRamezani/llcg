import os
import copy
import torch
import math
import numpy as np

def metis(dataset, num_parts):

    data = dataset[0]

    # Normalize the Adjacency
    # data = T.GCNNorm()(data)

    dist_dir = os.path.join(dataset.processed_dir, '../partitioned/')

    # save clustering info for later
    cluster_data = data.adj_t.partition(num_parts=num_parts)

    adj = cluster_data[0]
    partptr = cluster_data[1]
    perm = cluster_data[2]

    # if not os.path.exists(dist_dir):
    #     os.mkdir(dist_dir)
    # part_info = os.path.join(dist_dir, 'partition_{}.pt'.format(num_parts))
    # print(part_info)
    # # FIXME: I changed it, fix it later

    partitioned_dir = os.path.join(dist_dir, 'metis-{}'.format(num_parts))
    print(partitioned_dir)
    if not os.path.exists(partitioned_dir):
        os.mkdir(partitioned_dir)

    torch.save((perm, partptr), partitioned_dir+'/perm.pt')

    start = partptr[0]
    part_cnt = 0
    for end in partptr[1:]:

        # Adj Partitioning
        part_adj = []
        adj_pbn = adj.narrow(0, start, end-start)
        i = partptr[0]
        for j in partptr[1:]:
            adj_pbp = adj_pbn.narrow(1, i, j-i)
            part_adj.append(adj_pbp)
            i = j
        torch.save(part_adj, partitioned_dir+'/adj_{}.pt'.format(part_cnt))

        # Features and Labels Partitioning
        part_feats = data.x[perm[start:end]]
        part_labels = data.y[perm[start:end]]
        part_train_mask = data.train_mask[perm[start:end]]
        part_val_mask = data.val_mask[perm[start:end]]
        part_test_mask = data.test_mask[perm[start:end]]
        torch.save((part_feats, part_labels, part_train_mask, part_val_mask, part_test_mask),
                   partitioned_dir+'/fela_{}.pt'.format(part_cnt))

        start = end
        part_cnt += 1

    # TODO: save meta info for the dataset to avoid loading everything