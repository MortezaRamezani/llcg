import os
import copy
import time
import torch
import torchmetrics
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from tqdm import trange
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from ogb.nodeproppred import Evaluator

from ..utils import Stats
from ..models import model_selector
from ..layers import layer_selector


class Base(object):

    def __init__(self,
                 config,
                 dataset
                 ):

        self.config = config
        self.stats = Stats(config)
        self.dataset = dataset

        # Set device
        if self.config.gpu >= 0:
            self.device = self.config.gpu
        else:
            self.device = 'cpu'
        
        if self.config.cpu_val:
            self.val_device = 'cpu'
        else:
            self.val_device = self.device

        model = model_selector(self.config.model)
        layer = layer_selector(self.config.layer)

        self.activation = torch.nn.ReLU(True)

        self.model = model(
            dataset.num_features,
            self.config.hidden_size,
            dataset.num_classes,
            self.config.num_layers,
            self.activation,
            layer=layer,
            input_norm=self.config.input_norm,
            layer_norm=self.config.layer_norm,
            arch=self.config.arch,
            residual=self.config.residual,
            dropout=self.config.dropout,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        if config.loss == 'xentropy':
            self.loss_fnc = torch.nn.CrossEntropyLoss()
        elif config.loss == 'bceloss':
            self.loss_fnc = torch.nn.BCEWithLogitsLoss()

        self.train_output = None

        if self.dataset.name.startswith('ogbn'):
            self.evaluator = Evaluator(name=self.dataset.name)

    def run(self):

        self.model = self.model.to(self.device)

        self.tbar = trange(self.config.num_epochs, desc='Epochs')
        

        # for epoch in range(self.config.num_epochs):
        for epoch in self.tbar:
            start_time = time.perf_counter()
            
            self.train(epoch)
            end_train_time = time.perf_counter()
            
            if not self.validation(epoch):
                break
            end_val_time = time.perf_counter()

            self.tbar.set_postfix(loss=f'{self.stats.train_loss[-1]:.4f}' if len(self.stats.train_loss) > 0 else '-',
                                  score=f'{self.stats.train_scores[-1]*100:.2f}' if len(self.stats.train_scores) > 0 else '-',
                                  val_loss=f'{self.stats.val_loss[-1]:.4f}',
                                  val_score=f'{self.stats.val_scores[-1]*100:.2f}'
                                  )

            self.stats.train_time.append(end_train_time-start_time)
            self.stats.val_time.append(end_val_time-end_train_time)

        self.inference()
        end_inf_time = time.perf_counter()
        self.stats.test_time = end_inf_time - end_val_time

        print(f'Test Score: {self.stats.test_score * 100:.2f}% '
              f'@ Epoch #{self.stats.best_val_epoch}, '
              f'Highest Val: {self.stats.best_val_score * 100:.2f}%'
              )

        print(f'Total training time: {np.sum(self.stats.train_time):.3f} sec.')

    def calc_score(self, pred, batch_labels):

        # Same device (GPU) metrics
        if self.dataset.name == 'ogbn-proteins':
            # ROC
            pred_labels = torch.nn.Sigmoid()(pred)
            score = torchmetrics.functional.auroc(pred_labels, batch_labels.int(),
                                                  num_classes=batch_labels.shape[1]).cpu().item()
        elif self.dataset.name == 'yelp':
            pred_labels = torch.nn.Sigmoid()(pred)
            score = torchmetrics.functional.f1(pred_labels, batch_labels.int()).cpu().item()
        else:
            pred_labels = pred.argmax(dim=-1, keepdim=True)
            # score = torchmetrics.functional.f1(pred_labels, batch_labels.int()).cpu().item()
            score = pred_labels.eq(batch_labels.unsqueeze(-1)).sum().cpu().item() / batch_labels.size(0)

        return score

    def patience(self):
        # terminate after num_patience of not increasing val_score
        if len(self.stats.val_scores) > self.config.val_patience and \
                np.max(self.stats.val_scores[-1*self.config.val_patience:]) < self.stats.best_val_score:
            print('Run out of patience!')
            return False

        return True

    def save(self, *args, **kwargs):
        self.stats.save()

    def train(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def validation(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def inference(self):
        raise NotImplementedError


# if self.dataset.name.startswith('ogbn'):
#     # Default OGB, on CPU using Numpy
#     if self.dataset.name == 'ogbn-proteins':
#         pred_labels = pred
#         score = self.evaluator.eval({
#             'y_true': batch_labels,
#             'y_pred': pred_labels,
#         })['rocauc']
#     else:
#         pred_labels = pred.detach().argmax(dim=-1, keepdim=True)
#         score = self.evaluator.eval({
#             'y_true': batch_labels.unsqueeze(-1),
#             'y_pred': pred_labels,
#         })['acc']
# elif self.dataset.name == 'yelp':
#     pred_labels = torch.nn.Sigmoid()(pred).detach().cpu() > 0.5
#     score = f1_score(batch_labels.cpu(), pred_labels, average='micro')
#  else:
#     pred_labels = pred.detach().cpu().argmax(dim=1)
#     score = f1_score(batch_labels.cpu(), pred_labels, average='micro')

# TODO: https://github.com/tqdm/tqdm/issues/630, https://github.com/KimythAnly/qqdm/
# from qqdm import qqdm, format_str
# self.tbar = qqdm(range(self.config.num_epochs), desc=format_str('bold', 'Training'))
# self.tbar.set_infos({
#                 'loss': f'{train_loss:.4f}',
#                 'score': f'{train_score*100:.2f}',
#                 'val_loss': f'{self.stats.val_loss[-1]:.4f}',
#                 'val_score': f'{self.stats.val_scores[-1]*100:.2f}',
#             })