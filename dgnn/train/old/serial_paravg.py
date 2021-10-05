
import os
import copy
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

from . import Distributed

class SerializedParamsAvg(Distributed):
    """
    This class is only for testing purpose. 
    The calculation is done on single GPU/CPU and in serial mode.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.dataset = dataset[0]
        self.dataset = T.GCNNorm()(self.dataset)
        
        self.world_size = self.config.num_procs

        self.device = H.rank2dev(0, self.num_gpus)
        # self.device = H.rank2dev(0, 0)

        self.full_adj = self.dataset.adj_t.to(self.device)
        self.full_features = self.dataset.x.to(self.device)
        self.full_labels = self.dataset.y.to(self.device)
        self.full_train_mask = self.dataset.train_mask
        self.full_val_mask = self.dataset.val_mask
        self.full_test_mask = self.dataset.test_mask

        self.loss_fnc = torch.nn.CrossEntropyLoss()

    def start(self):
        self.train(0)


    def train(self, _, *args, **kwargs):
    
            
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

        total_train_size = np.sum(train_sizes)

        self.model = self.model.to(device)
        params = self.model.state_dict()
        server_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        for epoch in range(self.config.num_epochs):
            for rank in range(self.world_size):

                #! Communication, load from the server
                if epoch > 0:
                    if self.config.sync_local or epoch % self.config.local_updates == 0:
                        # if rank == 0:
                        #     print('Sync clients...')
                        client_models[rank].load_state_dict(params)
                        # client_optimizers[rank] = torch.optim.Adam(client_models[rank].parameters(), lr=self.config.lr)

                # train on clients
                client_models[rank].train()
                client_optimizers[rank].zero_grad()

                output = client_models[rank](features[rank], adjs[rank])
                loss = self.loss_fnc(output[train_masks[rank]], labels[rank][train_masks[rank]])

                loss.backward()
                # client_optimizers[rank].step()

            
            ## Attempt 1: get the fedavg params on servers
            # for layer in params.keys():
            #     tmp = []
            #     for rank in range(self.world_size):
            #         tmp.append(client_models[rank].state_dict()[layer])
            #     params[layer] = torch.div(torch.stack(tmp, dim=0).sum(dim=0), self.world_size)
            
            ## Attempt 2
            # for k in params.keys():
            #     params[k] = torch.zeros_like(params[k])

            # for rank in range(self.world_size):
            #     for k in params:
            #         # params[k] += torch.div(client_models[rank].state_dict()[k], self.world_size)
            #         params[k] += torch.div(client_models[rank].state_dict()[k] * train_sizes[rank], total_train_size)
            
            # Attempt 3
            server_model = self.model
            for sp, cp in zip(server_model.parameters(), client_models[0].parameters()):
                sp.grad = torch.div(cp.grad , total_train_size/train_sizes[0])

            for rank in range(1, self.world_size):
                for sp, cp in zip(server_model.parameters(), client_models[rank].parameters()):
                    sp.grad += torch.div(cp.grad , total_train_size/train_sizes[rank])
            server_optimizer.step()
            params = server_model.state_dict()

            val_score, val_loss = self.validation(params, epoch)
            print(f'Training Epoch #{epoch}, val score {val_score*100:.2f}, val loss {val_loss:.4f}')

        
        test_score = self.inference(params)
        print(f'Test accuracy is {test_score*100:.2f} at epoch {self.stats.best_val_epoch}')



    def validation(self, params, epoch):
        
        self.model.load_state_dict(params)
        model = self.model.to(self.device)
        
        model.eval()
        val_output = model(self.full_features, self.full_adj)

        val_loss = self.loss_fnc(val_output[self.full_val_mask], self.full_labels[self.full_val_mask])
        
        val_pred = val_output[self.full_val_mask].detach().argmax(dim=1)

        val_score = (val_pred.eq(
                self.full_labels[self.full_val_mask]).sum() / self.full_val_mask.sum()).item()


        if self.stats.best_val_score == 0 or val_score > self.stats.best_val_score:
            self.stats.best_val_score = val_score
            self.stats.best_model = copy.deepcopy(model)
            self.stats.best_val_epoch = epoch

        self.stats.val_scores.append(val_score)

        return val_score, val_loss

    def inference(self, params):
        # self.model.load_state_dict(params)
        # model = self.model.to(self.device)
        model = self.stats.best_model.to(self.device)

        test_output = model(self.full_features, self.full_adj)
        test_pred = test_output[self.full_test_mask].argmax(dim=1)

        test_score = (test_pred.eq(
                self.full_labels[self.full_test_mask]).sum() / self.full_test_mask.sum()).item()
        
        self.stats.test_score = test_score
        
        return test_score