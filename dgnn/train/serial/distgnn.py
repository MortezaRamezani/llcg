
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

from .distgnn_full import DistGNNFull


class DistGNN(DistGNNFull):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.clients_trainloader = []

        for rank in range(self.config.world_size):
            
            tmp_adj = self.clients_adj[rank]

            if self.config.sampler == 'subgraph':
                tmp_train_loader = samplers.SubGraphSampler(tmp_adj,
                                                            self.config.minibatch_size,
                                                            num_workers=self.config.num_samplers,
                                                            num_batches=self.config.local_updates,
                                                            num_layers=self.config.num_layers,
                                                            persistent_workers=True,
                                                            )
            elif config.sampler == 'neighbor':
                tmp_train_loader = samplers.NeighborSampler(tmp_adj,
                                                            self.config.minibatch_size,
                                                            num_workers=self.config.num_samplers,
                                                            num_batches=self.config.local_updates,
                                                            num_layers=self.config.num_layers,
                                                            num_neighbors=self.config.num_neighbors,
                                                            persistent_workers=True,
                                                            )

            self.clients_trainloader.append(tmp_train_loader)

        if self.config.sampler == 'neighbor':
            print('FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU')
            self.full_adj = row_norm(self.dataset[0].adj_t.to_symmetric()).to(self.device)

    def local_train(self, rank, epoch):

        # To speedup serial training, move all features and labels
        if epoch == 0:
            print('Local Train', rank, len(self.clients_trainloader[rank]))
            self.clients_features[rank] = self.clients_features[rank].to(self.device)
            self.clients_labels[rank] = self.clients_labels[rank].to(self.device)

        self.client_sync(rank)

        self.clients_model[rank].train()

        for input_nid, nodeblocks, output_nid in self.clients_trainloader[rank]:

            nodeblocks.to(self.device)

            features = self.clients_features[rank][input_nid] #.to(self.device)
            labels = self.clients_labels[rank][output_nid] #.to(self.device)
            train_mask = self.clients_train_mask[rank][output_nid]

            # import pdb; pdb.set_trace()

            self.clients_optimizer[rank].zero_grad()

            output = self.clients_model[rank](features, nodeblocks)
            loss = self.loss_fnc(output[train_mask], labels[train_mask])

            loss.backward()
            self.clients_optimizer[rank].step()

            if not self.config.weight_avg:
                for i, cp in enumerate(self.clients_model[rank].parameters()):
                    self.clients_grads[rank][i] += cp.grad
