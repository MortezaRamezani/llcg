import os
import copy
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from multiprocessing import Value
from ctypes import c_bool
from tqdm import trange

from ..base import Base
from .distgnn import DistGNN
from ...data import samplers, partition, Dataset
from ...utils import helpers as H
from ...data.transforms import row_norm


class DistDGL(DistGNN):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # self.raw_adj = self.dataset[0].adj_t


    # worker training
    @staticmethod
    def workers(rank, params_queue, ready_flag, config, dataset_rawdir,
                global_model, loss_fnc, meta_queue, end_train, comm_cost):

        ready_flag = ready_flag[rank]

        dataset_dir = os.path.join(dataset_rawdir[:-3], config.partitioned_dir)
        dataset_processed = os.path.join(dataset_rawdir[:-3], config.processed_filename)

        # part_idx for metis loaded seperately
        perm, part_ptr = torch.load(dataset_dir+'/perm.pt')
        start = rank
        end = rank + 1
        part_idx = perm[part_ptr[start]:part_ptr[end]]
        
        # open full adj and features and lab and mask
        dataset = torch.load(dataset_processed)

        adj = dataset[0].adj_t
        if type(adj) == list:
            adj = adj[0]

        feat = dataset[0].x
        lab = dataset[0].y
        full_train_mask = dataset[0].train_mask

        tr_mask = full_train_mask[part_idx]

        # if rank == 0:
        #     print(dataset_dir, dataset_processed)
        #     print(adj, feat, lab, part_idx.shape, tr_mask.shape)

        meta_queue.put((rank, tr_mask.count_nonzero()))

        device = H.rank2dev(rank, config.num_gpus)

        if config.sampler == 'neighbor':
            train_loader = samplers.NeighborSampler(adj,
                                                    config.minibatch_size,
                                                    num_workers=config.num_samplers,
                                                    num_batches=config.local_updates,
                                                    num_layers=config.num_layers,
                                                    num_neighbors=config.num_neighbors,
                                                    minibatch='dglsim',
                                                    node_idx=part_idx,
                                                    persistent_workers=True,
                                                    )
        else:
            raise NotImplementedError

        model = copy.deepcopy(global_model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        model_size = 0
        params = model.state_dict()
        for k in params:
            model_size += params[k].element_size() * params[k].nelement()

        feat_size = feat[0].element_size() * feat[0].nelement()


        for epoch in range(config.num_epochs):

            feat_cost = 0

            if end_train.value:
                break

            if epoch > 0:
                # Sync with Param Server
                model.load_state_dict(global_model.state_dict())

            model.train()
            ready_flag.clear()

            # Train Locally for K iterations
            for input_nid, nodeblocks, output_nid in train_loader:
                nodeblocks.to(device)
                features = feat[input_nid].to(device)
                labels = lab[output_nid].to(device)
                train_mask = full_train_mask[output_nid].to(device)

                # compute remote nodes cost
                diff = input_nid[~input_nid.unsqueeze(1).eq(part_idx).any(1)]

                # if rank == 0:
                #     print(part_idx)
                #     print(input_nid)
                #     print(diff)

                if diff.shape[0] > 0:
                    feat_cost += diff.shape[0] * feat_size

                optimizer.zero_grad()
                output = model(features, nodeblocks)
                loss = loss_fnc(output[train_mask], labels[train_mask])
                loss.backward()
                optimizer.step()

            # Move to CPU and put on the Queue
            params_dict = {}
            tmp_params = model.state_dict()
            for key in tmp_params:
                params_dict[key] = tmp_params[key].clone().cpu()
            
            params_queue.put(params_dict)

            # Cost of communication
            comm_cost[epoch] = model_size + feat_cost

            # Wait for server to continue with new global_model
            ready_flag.wait()

