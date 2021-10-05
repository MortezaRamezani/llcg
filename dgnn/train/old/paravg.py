
import os
import copy

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from ..utils import Stats
from ..models import model_selector
from ..data import partition as P
from ..utils import helpers as H

from . import Distributed

class ParamsAvg(Distributed):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.dataset = dataset[0]
        self.dataset = T.GCNNorm()(self.dataset)

    def train(self, rank, *args, **kwargs):
    
        # Load the rank-th partition, to rank-th device both adj and features
        adj, features, labels, train_mask, val_mask, test_mask = P.load_partitions(
            self.dataset_dir, rank)

        # Init the local model,
        model = copy.deepcopy(self.model)
        # model.update_rank(rank)
        device = H.rank2dev(rank, self.num_gpus)

        print(device, flush=True)

        # renormalize this rank adjacency again
        adj = adj[rank]
        # adj.storage._value = None
        adj.set_value(None)
        adj = gcn_norm(adj)

        model = model.to(device)
        adj = adj.to(device)
        features = features.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        labels = labels.to(device)

        loss_fnc = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

        best_model = copy.deepcopy(model)
        best_val_score = 0

        if rank == 0:
            full_adj = self.dataset.adj_t.to(device)
            full_features = self.dataset.x.to(device)
            full_labels = self.dataset.y.to(device)
            full_val_mask = self.dataset.val_mask

        for epoch in range(self.config.num_epochs):
            # train
            model.train()
            optimizer.zero_grad()

            output = model(features, adj)
            loss = loss_fnc(output[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

            # Fed Average
            # print(model.state_dict())
            # print(len(model.state_dict()))
            
            params = copy.deepcopy(model.state_dict())
            
            for layer in params.keys():
                dist.all_reduce(params[layer], op=dist.ReduceOp.SUM)
                # print('pa', params[layer].numel() * params[layer].element_size())
                params[layer] = torch.div(params[layer], self.config.num_procs)
            
            model.load_state_dict(params)

            # End of Epoch
            if rank == 0:
                self.stats.train_loss.append(loss.item())

                model.eval()
                val_output = model(full_features, full_adj)
                val_pred = val_output[full_val_mask].argmax(dim=1)
                val_score = (val_pred.eq(
                        full_labels[full_val_mask]).sum() / full_val_mask.sum()).item()
                self.stats.val_scores.append(val_score)

                print(f'Epoch  #{epoch}:',
                      f'train loss {loss.item():.3f}',
                      f'val accuracy {val_score*100:.2f}%',
                      flush=True)

        
        # Testing
        if rank == 0:
            print('End of training on rank 0, testing on full graph')
            # print(self.dataset)
            # adj = self.dataset.adj_t.to(device)
            # features = self.dataset.x.to(device)
            # labels = self.dataset.y.to(device)
            test_mask = self.dataset.test_mask

            model.eval()
            test_output = model(full_features, full_adj)
            test_pred = test_output[test_mask].argmax(dim=1)

            test_score = (test_pred.eq(
                full_labels[test_mask]).sum() / test_mask.sum()).item()

            print('Test accuracy is {:.2f}'.format(test_score*100))


        