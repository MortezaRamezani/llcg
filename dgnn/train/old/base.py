
import os
import copy

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from ...utils import Stats
from ...models import model_selector
from ...layers import layer_selector
from ...data import partition as P
from ...utils import helpers as H


class Base(object):
    """Base class for training GNN, single GPU (Process)

    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self,
                 config,
                 dataset,
                 ):

        # self.dataset = dataset
        self.config = config
        self.stats = Stats(config)
        self.dataset = dataset[0]
        self.dataset = T.GCNNorm()(self.dataset)

        # Set device
        if self.config.gpu >= 0:
            self.device = self.config.gpu
        else:
            self.device = 'cpu'

        model = model_selector(self.config.model)
        layer = layer_selector(self.config.layer)

        self.activation = torch.nn.ReLU(True)

        self.model = model(
            dataset.num_features,
            self.config.hidden_size,
            dataset.num_classes,
            self.config.num_layers,
            self.activation,
            layer=layer
        )

        self.model = self.model.to(self.device)

        self.loss_fnc = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0)



    def start(self):
        full_adj = self.dataset.adj_t.to(self.device)
        full_features = self.dataset.x.to(self.device)
        full_labels = self.dataset.y.to(self.device)

        full_train_mask, val_mask, test_mask = self.dataset.train_mask, self.dataset.val_mask, self.dataset.test_mask

        for epoch in range(self.config.num_epochs):

            if self.config.use_sampling:
                sampled_nodes = torch.randint(0, self.dataset.num_nodes, (self.config.minibatch_size, ), dtype=torch.long)
                sampled_nodes, _ = torch.sort(sampled_nodes)
                sampled_adj, _ = self.dataset.adj_t.saint_subgraph(sampled_nodes)
                sampled_feat = self.dataset.x[sampled_nodes]
                sampled_label = self.dataset.y[sampled_nodes]
                sampled_train_mask = self.dataset.train_mask[sampled_nodes]
                
                sampled_adj = sampled_adj.set_value(None)

                sampled_adj = gcn_norm(sampled_adj)
                adj = sampled_adj.to(self.device)
                features = sampled_feat.to(self.device)
                labels = sampled_label.to(self.device)
                train_mask = sampled_train_mask

            else:
                adj = full_adj
                features = full_features
                labels = full_labels
                train_mask = full_train_mask

            self.model.train()
            self.optimizer.zero_grad()

            output = self.model(features, adj)
            loss = self.loss_fnc(output[train_mask], labels[train_mask])

            loss.backward()
            self.optimizer.step()

            # Validation
            if not self.config.use_sampling:
                val_loss = self.loss_fnc(output[val_mask],labels[val_mask])
                val_pred = output[val_mask].detach().argmax(dim=1)
                val_score = (val_pred.eq(
                    labels[val_mask]).sum() / val_mask.sum()).item()
            else:
                self.model.eval()
                val_output = self.model(full_features, full_adj)
                val_loss = self.loss_fnc(val_output[val_mask], full_labels[val_mask])
                val_pred = val_output[val_mask].detach().argmax(dim=1)
                val_score = (val_pred.eq(
                    full_labels[val_mask]).sum() / val_mask.sum()).item()

            self.stats.val_scores.append(val_score)
            if val_score > self.stats.best_val_score:
                self.stats.best_val_epoch = epoch
                self.stats.best_val_score = val_score
                self.stats.best_model = copy.deepcopy(self.model)

            print(f'Epoch #{epoch}, train loss {loss:.2f} and val score {val_score*100:.2f}, val loss: {val_loss:.4f}')


        # testing
        self.stats.best_model.eval()
        test_output = self.stats.best_model(full_features, full_adj)
        test_pred = test_output[test_mask].argmax(dim=1)

        test_score = (test_pred.eq(
            full_labels[test_mask]).sum() / test_mask.sum()).item()

        self.stats.test_score = test_score

        print('Test accuracy is {:.2f}'.format(test_score*100))
            
    def validation(self):
        pass

    def inference(self):
        pass

    def save(self, *args, **kwargs):
        self.stats.save()
