
import os
import copy
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np

from ..utils import Stats
from ..models import model_selector
from ..data import partition as P
from ..utils import helpers as H

from . import SerializedParamsAvg

class SerializedFedAvgCorrection(SerializedParamsAvg):
    """
    This class is only for testing purpose. 
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.model = self.model.to(self.device)

        if self.config.full_correct:
            print('Full for correction!')

    def train(self, _, *args, **kwargs):
            
        adjs = []
        features = []
        labels = []
        train_masks = []
        train_sizes = []

        client_models = []
        client_optimizers = []
        client_grads = []

        device = self.device

        # Load all partitions, 
        for rank in range(self.world_size):

            # device = H.rank2dev(rank, self.num_gpus)
            adj, feat, lab, tr, va, te = P.load_partitions(self.dataset_dir, rank)
            adj = adj[rank]
            adj = adj.set_value(None)
            adj = gcn_norm(adj)
            adj = adj.to(device)

            feat = feat.to(device)
            lab = lab.to(device)

            model = copy.deepcopy(self.model)
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr, momentum=0)

            adjs.append(adj)
            features.append(feat)
            labels.append(lab)
            train_masks.append(tr)
            train_sizes.append(tr.count_nonzero())

            client_models.append(model)
            client_optimizers.append(optimizer)

            tmp_cgrad = []
            for p in model.parameters():
                tmp_cgrad.append(torch.zeros_like(p))
            client_grads.append(tmp_cgrad)

        total_train_size = np.sum(train_sizes)

        self.model = self.model.to(device)
        params = self.model.state_dict()
        server_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        for epoch in range(self.config.num_epochs):

            for inner_epoch in range(self.config.local_updates):
                for rank in range(self.world_size):
                    #! Communication, load from the server
                    if inner_epoch == 0:
                            if rank == 0:
                                print('Sync clients...')
                            client_models[rank].load_state_dict(params)

                            for i, cp in enumerate(client_models[rank].parameters()):
                                client_grads[rank][i] = torch.zeros_like(cp)

                    # train on clients
                    client_models[rank].train()
                    client_optimizers[rank].zero_grad()

                    output = client_models[rank](features[rank], adjs[rank])
                    loss = self.loss_fnc(output[train_masks[rank]], labels[rank][train_masks[rank]])

                    loss.backward()
                    client_optimizers[rank].step()

                    for i, cp in enumerate(client_models[rank].parameters()):
                        client_grads[rank][i] += cp.grad

            
            # Attempt 3
            server_model = self.model
            for sp, cp in zip(server_model.parameters(), client_models[0].parameters()):
                sp.grad = torch.zeros_like(cp.grad)

            for rank in range(0, self.world_size):
                for i, sp in enumerate(server_model.parameters()):
                    sp.grad += torch.div(client_grads[rank][i] , total_train_size/train_sizes[rank])

            server_optimizer.step()
            params = server_model.state_dict()


            if epoch != 0:

                print(f'Doing server pass for {self.config.server_epochs} epoch...')
                for server_epoch in range(self.config.server_epochs):

                    if not self.config.full_correct:
            
                        if not self.config.stratified:
                            sampled_nodes = torch.randint(0, self.dataset.num_nodes, (self.config.minibatch_size, ), dtype=torch.long)
                        else:
                            sampled_nodes = []
                            perm, partptr = P.load_meta(self.dataset_dir)
                            start = 0
                            num_samples = self.config.minibatch_size // self.config.num_procs
                            for end in partptr[1:]:
                                sampled_perm = torch.randperm(end - start)
                                if (end - start) < num_samples:
                                    sampled_idx = sampled_perm
                                else:
                                    sampled_idx = sampled_perm[:num_samples]
                                sampled_nodes.append(perm[sampled_idx])
                            
                            sampled_nodes = torch.cat(sampled_nodes)

                        # Temp Fix for subgraph bug
                        sampled_nodes, _ = torch.sort(sampled_nodes)
                        sampled_adj, _ = self.dataset.adj_t.saint_subgraph(sampled_nodes)
                        sampled_feat = self.dataset.x[sampled_nodes]
                        sampled_label = self.dataset.y[sampled_nodes]
                        sampled_train_mask = self.dataset.train_mask[sampled_nodes]
                        
                        sampled_adj = sampled_adj.set_value(None)

                        sampled_adj = gcn_norm(sampled_adj)
                        sampled_adj = sampled_adj.to(device)
                        sampled_feat = sampled_feat.to(device)
                        sampled_label = sampled_label.to(device)
                        
                    else:
                        sampled_adj = self.full_adj
                        sampled_feat = self.full_features
                        sampled_label = self.full_labels
                        sampled_train_mask = self.full_train_mask

                    self.model.load_state_dict(params)
                    self.model.train()
                    self.optimizer.zero_grad()

                    server_output = self.model(sampled_feat, sampled_adj)
                    server_loss = self.loss_fnc(server_output[sampled_train_mask], sampled_label[sampled_train_mask])
                    
                    server_loss.backward()
                    self.optimizer.step()

                params = self.model.state_dict()


            val_score, val_loss = self.validation(params, epoch)
            print(f'Training Epoch #{epoch}, val score {val_score*100:.2f}, val loss {val_loss:.4f}')

        
        test_score = self.inference(params)
        print(f'Test accuracy is {test_score*100:.2f} at epoch {self.stats.best_val_epoch}')
