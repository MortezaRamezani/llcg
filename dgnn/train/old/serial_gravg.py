
import os
import copy
from collections import OrderedDict

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor

from ..utils import Stats
from ..models import model_selector
from ..data import partition as P
from ..utils import helpers as H

from . import SerializedParamsAvg

import pdb

class MyLinear(torch.nn.Module):

    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(input_dim)/input_dim)

    def forward(self, x):
        w_normalized = torch.nn.functional.softmax(self.weight)
        output = torch.nn.functional.linear(x, w_normalized)

        return output


class GraAvg(torch.nn.Module):

    def __init__(self, num_layers, num_parts):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers
        self.activation = torch.nn.ReLU(True)

        for layer in range(num_layers):
            # self.layers.append(torch.nn.Linear(num_parts, 1, bias=False))
            # self.layers[layer].weight.data = torch.ones(num_parts) / num_parts
            self.layers.append(MyLinear(num_parts, 1))


    def forward(self, adj, x, stacked_params, return_params=False):
        # Compute AXWW', where A,X and W are input and W' is learnable
        h = x
        params = []
        for i, layer in enumerate(self.layers):
            ax = adj.spmm(h)
            ww = layer(stacked_params[i][1])
            if return_params:
                params.append(ww)
            h = torch.matmul(ax, ww.t())
            if i < self.num_layers - 1:
                h = self.activation(h)
        
        return h, params


class SerializedGrAvg(SerializedParamsAvg):
    """
    This class is only for testing purpose. 
    The calculation is done on single GPU/CPU and in serial mode.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.gravg_model = GraAvg(self.config.num_layers, self.config.num_procs)
        self.gravg_model = self.gravg_model.cuda()
        self.gravg_opt = torch.optim.Adam(self.gravg_model.parameters(), lr=self.config.lr)



    def train(self, rank, *args, **kwargs):
    
        loss_fnc = torch.nn.CrossEntropyLoss()
         
        adjs = []
        features = []
        labels = []
        train_masks = []
        
        client_models = []
        client_optimizers = []

        device = self.device

        # Load all partitions, 
        for rank in range(self.world_size):


            adj, feat, lab, tr, va, te = P.load_partitions(self.dataset_dir, rank)
            # device = H.rank2dev(rank, self.num_gpus) 

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

            client_models.append(model)
            client_optimizers.append(optimizer)

        params = OrderedDict()
        stacked_params = OrderedDict()
        for layer in self.model.state_dict().keys():
            params[layer] = None
            stacked_params[layer] = None
        
        for epoch in range(self.config.num_epochs):
            for rank in range(self.world_size):

                #! Communication, load from the server 
                if epoch > 0:
                    client_models[rank].load_state_dict(params)

                # train on clients
                client_models[rank].train()
                client_optimizers[rank].zero_grad()

                output = client_models[rank](features[rank], adjs[rank])
                loss = loss_fnc(output[train_masks[rank]], labels[rank][train_masks[rank]])

                loss.backward()
                client_optimizers[rank].step()

            
            # collect all params
            for layer in stacked_params.keys():
                tmp = []
                for rank in range(self.world_size):
                    tmp.append(client_models[rank].state_dict()[layer])
                stacked_params[layer] = torch.stack(tmp, dim=2)


            # sample a subgraph on server and train gravg model to get new params
            # pdb.set_trace()

            for se_ep in range(self.config.server_epochs):
                sampled_nodes = torch.randint(0, self.dataset.num_nodes, (self.config.minibatch_size, ), dtype=torch.long)
                # Temp fix for subgraph bug
                sampled_nodes, _ = torch.sort(sampled_nodes)
                sampled_adj, _ = self.dataset.adj_t.saint_subgraph(sampled_nodes)
                sampled_feat = self.dataset.x[sampled_nodes]
                sampled_label = self.dataset.y[sampled_nodes]
                sampled_train_mask = self.dataset.train_mask[sampled_nodes]
                sampled_adj = sampled_adj.set_value(None)
                
                # Temp fix for subgraph bug
                # tmp = sampled_adj.to_torch_sparse_coo_tensor()
                # sampled_adj = SparseTensor.from_torch_sparse_coo_tensor(tmp)
                
                # pdb.set_trace()
                sampled_adj = gcn_norm(sampled_adj)
                sampled_adj = sampled_adj.to(device)
                sampled_feat = sampled_feat.to(device)
                sampled_label = sampled_label.to(device)

                self.gravg_model.train()
                self.gravg_opt.zero_grad()

                get_new_params = False
                if se_ep == self.config.server_epochs - 1:
                    get_new_params = True

                server_output, new_params = self.gravg_model(sampled_adj, sampled_feat, list(stacked_params.items()), get_new_params)
                server_loss = loss_fnc(server_output[sampled_train_mask], sampled_label[sampled_train_mask])
                server_loss.backward()
                self.gravg_opt.step()


            # pdb.set_trace()

            # update params using gravg_model
            i = 0
            for layer in params.keys():
                params[layer] = new_params[i]
                i += 1


            val_score = self.validation(params, epoch)
            print(f'Training Epoch #{epoch}, val score {val_score*100:.2f}')

        
        test_score = self.inference(params)
        print(f'Test accuracy is {test_score*100:.2f} at epoch {self.stats.best_val_epoch}')