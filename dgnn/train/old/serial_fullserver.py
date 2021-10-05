
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

class SerializedFullServer(SerializedParamsAvg):
    """
    This class is only for testing purpose. 
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.model = self.model.to(self.device)

    def train(self, rank, *args, **kwargs):
    
        loss_fnc = torch.nn.CrossEntropyLoss()
            
        adjs = []
        features = []
        labels = []
        train_masks = []
        train_sizes = []

        client_models = []
        client_optimizers = []

        device = self.device

        # Load all partitions, 
        for rank in range(self.world_size):

            adj, feat, lab, tr, _, _ = P.load_partitions(self.dataset_dir, rank)

            adj = adj[rank]
            adj = adj.set_value(None)
            adj = gcn_norm(adj)
            adj = adj.to(device)

            feat = feat.to(device)
            lab = lab.to(device)

            model = copy.deepcopy(self.model)
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

            adjs.append(adj)
            features.append(feat)
            labels.append(lab)
            train_masks.append(tr)
            train_sizes.append(tr.count_nonzero())

            client_models.append(model)
            client_optimizers.append(optimizer)


        # params = OrderedDict()
        # for layer in self.model.state_dict().keys():
        #     params[layer] = None

        self.model = self.model.to(device)
        params = self.model.state_dict()
        total_train_size = np.sum(train_sizes)
        server_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        for epoch in range(self.config.num_epochs):
            # for rank in range(self.world_size):

            #     #! Communication, load from the server
            #     if epoch > 0:
            #         if self.config.sync_local or epoch % self.config.local_updates == 0:
            #             # if rank == 0:
            #             #     print('Sync clients...')
            #             client_models[rank].load_state_dict(params)

            #     # train on clients
            #     client_models[rank].train()
            #     client_optimizers[rank].zero_grad()

            #     output = client_models[rank](features[rank], adjs[rank])
            #     loss = loss_fnc(output[train_masks[rank]], labels[rank][train_masks[rank]])

            #     loss.backward()
            #     # client_optimizers[rank].step()

            
            # # get the fedavg params on servers
            # for layer in params.keys():
            #     tmp = []
            #     for rank in range(self.world_size):
            #         tmp.append(client_models[rank].state_dict()[layer])
            #     params[layer] = torch.div(torch.stack(tmp, dim=0).sum(dim=0), self.world_size)

            server_model = self.model
            # for sp, cp in zip(server_model.parameters(), client_models[0].parameters()):
            #     sp.grad = torch.div(cp.grad , total_train_size/train_sizes[0])

            # for rank in range(1, self.world_size):
            #     for sp, cp in zip(server_model.parameters(), client_models[rank].parameters()):
            #         sp.grad += torch.div(cp.grad , total_train_size/train_sizes[rank])
            # # server_optimizer = torch.optim.Adam(server_model.parameters(), lr=self.config.lr)
            # server_optimizer.step()
            params = server_model.state_dict()
            
            if epoch != 0 and epoch % self.config.local_updates == 0 or epoch == self.config.num_epochs - 1:

                print(f'Doing server pass for {self.config.server_epochs} epoch...')
                for server_epoch in range(self.config.server_epochs):

                    self.model.load_state_dict(params)

                    self.model.train()
                    self.optimizer.zero_grad()
                    server_output = self.model(self.full_features, self.full_adj)
                    server_loss = loss_fnc(server_output[self.full_train_mask], self.full_labels[self.full_train_mask])
                    server_loss.backward()
                    self.optimizer.step()

                params = self.model.state_dict()


            val_score, val_loss = self.validation(params, epoch)
            print(f'Training Epoch #{epoch}, val score {val_score*100:.2f}, val loss {val_loss:.4f}')

        
        test_score = self.inference(params)
        print(f'Test accuracy is {test_score*100:.2f} at epoch {self.stats.best_val_epoch}')

