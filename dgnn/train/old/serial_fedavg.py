
import os
import copy
import math
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np

from ...utils import Stats
from ...models import model_selector
from ...data import partition as P
from ...utils import helpers as H

from . import SerializedParamsAvg

class SerializedFedAvg(SerializedParamsAvg):
    """
    This class is only for testing purpose. 
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.model = self.model.to(self.device)

        self.parted = False
    
    def load_fixed_part(self, rank):

        if not self.parted:
            self.num_nodes = self.dataset.num_nodes
            self.node_per_part = math.ceil(self.num_nodes / self.config.num_procs)
            self.part_nodes = torch.split(torch.arange(self.num_nodes), self.node_per_part)

            self.part_ptr = [self.part_nodes[0][0]]
            for part in self.part_nodes:
                self.part_ptr.append(part[-1])

            self.parted = True


        start = self.part_ptr[rank]
        if start > 0:
            start += 1
        end = self.part_ptr[rank+1]
        adj = self.dataset.adj_t.narrow(0, start, end-start+1).narrow(1, start, end-start+1)


        part_feats = self.dataset.x[start:end+1]
        part_labels = self.dataset.y[start:end+1]
        part_train_mask = self.dataset.train_mask[start:end+1]
        part_val_mask = self.dataset.val_mask[start:end+1]
        part_test_mask = self.dataset.test_mask[start:end+1]

        return adj, part_feats, part_labels, part_train_mask, part_val_mask, part_test_mask

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
            if self.config.part_method == 'metis':
                adj, feat, lab, tr, va, te = P.load_partitions(self.dataset_dir, rank)
                adj = adj[rank]
            else:
                adj, feat, lab, tr, va, te = self.load_fixed_part(rank)

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

        # import pdb; pdb.set_trace()

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

            val_score, val_loss = self.validation(params, epoch)
            print(f'Training Epoch #{epoch}, val score {val_score*100:.2f}, val loss {val_loss:.4f}')

        
        test_score = self.inference(params)
        print(f'Test accuracy is {test_score*100:.2f} at epoch {self.stats.best_val_epoch}')
