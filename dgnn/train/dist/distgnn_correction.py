import os
import copy
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

from tqdm import trange
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from . import DistGNN
from ...data import samplers, partition
from ...utils import helpers as H


class DistGNNCorrection(DistGNN):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.dataset_dir = os.path.join(dataset.raw_dir[:-3], self.config.partitioned_dir)
        full_adj = self.dataset[0].adj_t
        
        # if self.dataset.name.startswith('ogbn'):
        #     full_adj = self.dataset[0].adj_t.to_symmetric()

        if self.config.server_sampler == 'subgraph':
            self.server_trainloader = samplers.SubGraphSampler(full_adj,
                                                               self.config.server_minibatch_size,
                                                               num_workers=self.config.server_num_samplers,
                                                               num_layers=self.config.num_layers,
                                                               num_batches=self.config.server_updates,
                                                               minibatch=self.config.server_minibatch,
                                                               part_meta=self.dataset_dir,
                                                               persistent_workers=True,
                                                               )
        elif config.server_sampler == 'neighbor':
            self.server_trainloader = samplers.NeighborSampler(full_adj,
                                                               self.config.minibatch_size,
                                                               num_workers=self.config.num_samplers,
                                                               num_layers=self.config.num_layers,
                                                               num_batches=self.config.server_updates,
                                                               num_neighbors=self.config.server_num_neighbors,
                                                               minibatch=self.config.server_minibatch,
                                                               part_meta=self.dataset_dir,
                                                               persistent_workers=True,
                                                               )


        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.server_lr)

    def server_average(self, local_params, epoch):
        if not self.config.server_opt_sync:
            super().server_average(local_params, epoch)
        else:
            if epoch == 0:
                print('OPT Sync enabled!')
            self.server_average_opt_sync(local_params, epoch)

        self.server_correction(epoch)

    def server_average_opt_sync(self, params_opt, epoch):

        params = self.global_model.state_dict()

        for k in params.keys():
            params[k] = torch.zeros_like(params[k], dtype=torch.float, device='cpu')

        for rank in range(self.config.world_size):
            for k in params:
                params[k] += torch.div(params_opt[rank][0][k] *
                                       self.workers_train_size[rank], self.workers_total_train)

        self.global_model.load_state_dict(params)

        ####################################################
        opt_state_dict = self.optimizer.state_dict()
        new_state = copy.deepcopy(params_opt[0][1])

        for k in new_state:
            for pg in new_state[k]:
                st_pg = new_state[k][pg]
                if torch.is_tensor(st_pg):
                    new_state[k][pg] = torch.zeros_like(st_pg, dtype=torch.float, device='cpu')
        

        for rank in range(self.config.world_size):
            for k in new_state:
                for pg in new_state[k]:
                    st_pg = new_state[k][pg]
                    if torch.is_tensor(st_pg):
                        new_state[k][pg] += torch.div(params_opt[rank][1][k][pg] *
                                       self.workers_train_size[rank], self.workers_total_train)
                    else:
                        new_state[k][pg] = params_opt[rank][1][k][pg]


        all_new_state = {'state': new_state, 'param_groups': opt_state_dict['param_groups']}
        self.optimizer.load_state_dict(all_new_state)


    def server_correction(self, epoch):
        
        start_ep = self.config.server_start_epoch

        if start_ep == -1:
            c_epoch = 0
            start_ep = self.config.num_epochs - 1

        if epoch < start_ep:
            return

        if epoch == start_ep:
            print('Server correction!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        self.model.load_state_dict(self.global_model.state_dict())

        self.model.train()

        for input_nid, nodeblocks, output_nid in self.server_trainloader:
    
            nodeblocks.to(self.val_device)
            features = self.full_features[input_nid]
            labels = self.full_labels[output_nid]
            train_mask = self.full_train_mask[output_nid]

            self.optimizer.zero_grad()
            output = self.model(features, nodeblocks)
            loss = self.loss_fnc(output[train_mask], labels[train_mask])
            loss.backward()
            self.optimizer.step()

            # print val score if it's correction at the end
            if self.config.server_start_epoch == -1:
                c_epoch += 1
                self.validation(epoch + c_epoch)
                print(f'Epoch Correction #{c_epoch}: '
                            f'val loss {self.stats.val_loss[-1]:.4f}, '
                            f'val f1 {self.stats.val_scores[-1]*100:.2f}.'
                    )
        
        self.global_model.load_state_dict(self.model.state_dict())
    
    # worker training
    @staticmethod
    def workers(rank, params_queue, ready_flag, config, dataset_rawdir,
                global_model, loss_fnc, meta_queue, end_train, comm_cost):

        ready_flag = ready_flag[rank]

        dataset_dir = os.path.join(dataset_rawdir[:-3], config.partitioned_dir)
        adj, feat, lab, tr_mask, _, _ = partition.load_partitions(dataset_dir, rank)

        meta_queue.put((rank, tr_mask.count_nonzero()))

        if config.part_method == 'metis':
            adj = adj[rank]

        # if config.dataset in ['arxiv']:
        #     adj = adj.to_symmetric()

        device = H.rank2dev(rank, config.num_gpus)

        if config.sampler == 'subgraph':
            train_loader = samplers.SubGraphSampler(adj,
                                                    config.minibatch_size,
                                                    num_workers=config.num_samplers,
                                                    num_batches=config.local_updates,
                                                    num_layers=config.num_layers,
                                                    persistent_workers=True,
                                                    )
        elif config.sampler == 'neighbor':
            train_loader = samplers.NeighborSampler(adj,
                                                    config.minibatch_size,
                                                    num_workers=config.num_samplers,
                                                    num_batches=config.local_updates,
                                                    num_layers=config.num_layers,
                                                    num_neighbors=config.num_neighbors,
                                                    persistent_workers=True,
                                                    )

        model = copy.deepcopy(global_model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        new_k = []
        if config.rho != 1:
            for i in range(config.num_epochs):
                new_k.append(int(np.ceil((config.local_updates / np.power(config.rho, i)))))
            if config.inc_k:
                new_k.reverse()

        if rank == 0:
            print('new_k', new_k)
        
        model_size = 0
        params = model.state_dict()
        for k in params:
            model_size += params[k].element_size() * params[k].nelement()

        for epoch in range(config.num_epochs):

            if end_train.value:
                break

            if epoch > 0:
                # Sync with Param Server
                model.load_state_dict(global_model.state_dict())

            model.train()
            ready_flag.clear()

            if config.rho != 1:
                # new_k = int(np.ceil((config.local_updates / np.power(config.rho, epoch))))
                train_loader.update_k(new_k[epoch])
                # if rank == 0:
                #     print('K updated to:', new_k)

            # Train Locally for K iterations
            for input_nid, nodeblocks, output_nid in train_loader:
                nodeblocks.to(device)
                features = feat[input_nid].to(device)
                labels = lab[output_nid].to(device)
                train_mask = tr_mask[output_nid].to(device)

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
            

            # Move optimizer state to queue as well
            if config.server_opt_sync:
                optim_state = {}
                tmp_opt_state = optimizer.state_dict()['state']
                for pg in tmp_opt_state:
                    pg_opt_state = {}
                    for state in tmp_opt_state[pg]:
                        state_val = tmp_opt_state[pg][state]
                        if torch.is_tensor(state_val):
                            pg_opt_state[state] = state_val.clone().cpu()
                        else:
                            pg_opt_state[state] = state_val
                    optim_state[pg] = pg_opt_state
                
                params_queue.put((params_dict, optim_state))
            else:
                # Normal
                params_queue.put(params_dict)

            # Cost of communication
            comm_cost[epoch] = model_size

            # Wait for server to continue with new global_model
            ready_flag.wait()