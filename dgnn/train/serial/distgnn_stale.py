import copy
import math
import numpy as np
import torch

from . import DistGNNCorr
from ...data import samplers


class DistGNNStale(DistGNNCorr):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.staled_model = None


    def server_average(self):
        self.staled_model = copy.deepcopy(self.model)
        DistGNNCorr.server_average(self)

    def server_correction(self):

        old_params = self.staled_model.state_dict()
        
        self.staled_model.train()

        for input_nid, nodeblocks, output_nid in self.server_trainloader:

            nodeblocks.to(self.device)
            features = self.full_features[input_nid]
            labels = self.full_labels[output_nid]
            train_mask = self.full_train_mask[output_nid]

            self.optimizer.zero_grad()
            output = self.staled_model(features, nodeblocks)
            loss = self.loss_fnc(output[train_mask], labels[train_mask])
            loss.backward()
            self.optimizer.step()

        updated_params = self.staled_model.state_dict()
        new_params = self.model.state_dict()

        for params in new_params:
            new_params[params] = new_params[params] + (updated_params[params] - old_params[params])

        self.model.load_state_dict(new_params)