
import copy
import math
import numpy as np
import torch

from . import DistGNN, DistGNNFull
from ...data import samplers
from ...data.transforms import row_norm

class DistGNNFullCorr(DistGNNFull):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # self.full_adj = self.dataset[0].adj_t
        # if self.dataset.name.startswith('ogbn'):
        #     self.full_adj = self.dataset[0].adj_t.to_symmetric()
        
        if self.config.sampler == 'subgraph':
            self.server_trainloader = samplers.SubGraphSampler(self.dataset[0].adj_t.to_symmetric(),
                                                        self.config.server_minibatch_size,
                                                        num_workers=self.config.num_samplers,
                                                        num_batches=self.config.server_updates,
                                                        num_layers=self.config.num_layers,
                                                        minibatch=self.config.server_minibatch,
                                                        part_meta=self.dataset_dir,
                                                        persistent_workers=True,
                                                        )
        elif config.sampler == 'neighbor':
            self.server_trainloader = samplers.NeighborSampler(self.dataset[0].adj_t.to_symmetric(),
                                                        self.config.server_minibatch_size,
                                                        num_workers=self.config.num_samplers,
                                                        num_batches=self.config.server_updates, 
                                                        num_layers=self.config.num_layers,
                                                        num_neighbors=self.config.server_num_neighbors,
                                                        minibatch=self.config.server_minibatch,
                                                        part_meta=self.dataset_dir,
                                                        persistent_workers=True,
                                                        )


    def train(self, epoch):

        self.model.train()

        for rank in range(self.config.world_size):
            self.local_train(rank, epoch)
        
        self.server_average()
        self.server_correction(epoch)

    def server_correction(self, epoch):
        
        self.model.train()

        for input_nid, nodeblocks, output_nid in self.server_trainloader:
            if epoch == 0:
                print('Server correction!', len(self.server_trainloader))

            nodeblocks.to(self.device)
            features = self.full_features[input_nid]
            labels = self.full_labels[output_nid]
            train_mask = self.full_train_mask[output_nid]
        
            self.optimizer.zero_grad()
            output = self.model(features, nodeblocks)
            loss = self.loss_fnc(output[train_mask], labels[train_mask])
            loss.backward()
            self.optimizer.step()

        # if epoch == 0:
        #     print('Server Correction!!')
        
        # nodeblocks = self.full_adj
        # features = self.full_features
        # labels = self.full_labels
        # train_mask = self.full_train_mask

        # self.optimizer.zero_grad()
        # output = self.model(features, nodeblocks)
        # loss = self.loss_fnc(output[train_mask], labels[train_mask])
        # loss.backward()
        # self.optimizer.step()