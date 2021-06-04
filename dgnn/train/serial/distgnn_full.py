import os
import copy
import math
import numpy as np
import torch

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from ..base import Base
from ..full import Full
from ...data.transforms import row_norm
from ...data import samplers, partition

class DistGNNFull(Full, Base):

    def __init__(self, config, dataset):

        # if full inference this, else Base init
        Full.__init__(self, config, dataset)

        self.dataset_dir = os.path.join(dataset.raw_dir[:-3], self.config.partitioned_dir)

        self.clients_adj = []
        self.clients_features = []
        self.clients_labels = []
        self.clients_train_mask = []
        self.clients_train_sizes = []

        self.clients_model = []
        self.clients_optimizer = []
        self.clients_grads = []

        for rank in range(self.config.world_size):
            # load partitions
            tmp_adj, tmp_feat, tmp_lab, tmp_tr, _, _ = partition.load_partitions(self.dataset_dir, rank)

            if self.config.part_method == 'metis':
                # import pdb; pdb.set_trace()
                tmp_adj = tmp_adj[rank]

            if self.dataset.name.startswith('ogbn') and self.dataset.name != 'ogbn-proteins':
                tmp_adj = tmp_adj.to_symmetric()

            self.clients_adj.append(tmp_adj)
            self.clients_features.append(tmp_feat)
            self.clients_labels.append(tmp_lab)
            self.clients_train_mask.append(tmp_tr)
            self.clients_train_sizes.append(tmp_tr.count_nonzero())

            # model and grads...
            tmp_model = copy.deepcopy(self.model).to(self.device)
            tmp_opt = torch.optim.Adam(tmp_model.parameters(), lr=self.config.lr)
            tmp_grads = []

            for p in tmp_model.parameters():
                tmp_grads.append(torch.zeros_like(p))

            self.clients_model.append(tmp_model)
            self.clients_optimizer.append(tmp_opt)
            self.clients_grads.append(tmp_grads)

        self.total_train_size = np.sum(self.clients_train_sizes)

    def train(self, epoch):
        self.model.train()
        for rank in range(self.config.world_size):
            self.local_train(rank, epoch)
        self.server_average()

    def local_train(self, rank, epoch):

        if epoch == 0:
            print('Local Train', rank)
            self.clients_adj[rank] = gcn_norm(self.clients_adj[rank]).to(self.device)
            # self.clients_adj[rank] = row_norm(self.clients_adj[rank]).to(self.device)
            self.clients_features[rank] = self.clients_features[rank].to(self.device)
            self.clients_labels[rank] = self.clients_labels[rank].to(self.device)

            # import pdb; pdb.set_trace()

        self.client_sync(rank)

        self.clients_model[rank].train()

        adj = self.clients_adj[rank]
        features = self.clients_features[rank]
        labels = self.clients_labels[rank]
        train_mask = self.clients_train_mask[rank]

        for inner_epoch in range(self.config.local_updates):

            self.clients_optimizer[rank].zero_grad()

            output = self.clients_model[rank](features, adj)
            loss = self.loss_fnc(output[train_mask], labels[train_mask])

            loss.backward()
            self.clients_optimizer[rank].step()

            if not self.config.weight_avg:
                for i, cp in enumerate(self.clients_model[rank].parameters()):
                    self.clients_grads[rank][i] += cp.grad

    def client_sync(self, rank):
        # Sync
        self.clients_model[rank].load_state_dict(self.model.state_dict())

        if not self.config.weight_avg:
            for i, cp in enumerate(self.clients_model[rank].parameters()):
                self.clients_grads[rank][i] = torch.zeros_like(cp)

    def server_average(self):

        if self.config.weight_avg:
            params = self.model.state_dict()
            for k in params.keys():
                params[k] = torch.zeros_like(params[k], dtype=torch.float)

            for rank in range(self.config.world_size):
                for k in params:
                    params[k] += torch.div(self.clients_model[rank].state_dict()[k] *
                                           self.clients_train_sizes[rank], self.total_train_size)

            self.model.load_state_dict(params)

        else:
            print('Grad Agg')
            server_model = self.model
            for sp, cp in zip(server_model.parameters(), self.clients_model[0].parameters()):
                sp.grad = torch.zeros_like(cp.grad, dtype=torch.float)

            for rank in range(self.config.world_size):
                for i, sp in enumerate(server_model.parameters()):
                    sp.grad += torch.div(self.clients_grads[rank][i],
                                         self.total_train_size/self.clients_train_sizes[rank])

            self.optimizer.step()