
import copy
import math
import numpy as np
import torch

from . import DistGNN, DistGNNFull
from ...data import samplers
from ...data.transforms import row_norm


class DistGNNCorr(DistGNN):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        full_adj = self.dataset[0].adj_t
        if self.dataset.name.startswith('ogbn'):
            full_adj = self.dataset[0].adj_t.to_symmetric()

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

            # self.full_adj = row_norm(self.full_adj).to(self.device)

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
                print('Server correction!')

            nodeblocks.to(self.device)
            features = self.full_features[input_nid]
            labels = self.full_labels[output_nid]
            train_mask = self.full_train_mask[output_nid]

            self.optimizer.zero_grad()
            output = self.model(features, nodeblocks)
            loss = self.loss_fnc(output[train_mask], labels[train_mask])
            loss.backward()
            self.optimizer.step()
