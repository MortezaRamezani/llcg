import copy
import torch
import time

from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np

from .base import Base
from ..data.transforms import row_norm

from sklearn.preprocessing import StandardScaler

class Full(Base):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # check if full inference is set
        self.full_adj = self.dataset[0].adj_t

        # if self.dataset.name.startswith('ogbn') and self.dataset.name != 'ogbn-proteins':
        #     self.full_adj = self.full_adj.to_symmetric()

        self.full_adj = gcn_norm(self.full_adj)
        # self.full_adj = row_norm(self.full_adj)

        self.full_device = self.val_device if self.config.cpu_val else self.device


        # import pdb; pdb.set_trace()

        self.full_adj = self.full_adj.to(self.full_device)
        # self.full_features = self.dataset[0].x.to(self.full_device)
        self.full_labels = self.dataset[0].y.to(self.full_device)
        
        train_feats = self.dataset[0].x[self.dataset[0].train_mask]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        self.full_features = torch.FloatTensor(scaler.transform(self.dataset[0].x)).to(self.full_device)
        # import pdb; pdb.set_trace()

        self.full_train_mask = self.dataset[0].train_mask #.to(self.device)
        self.full_val_mask = self.dataset[0].val_mask
        self.full_test_mask = self.dataset[0].test_mask

    def train(self, epoch):

        self.model.train()

        adj = self.full_adj
        features = self.full_features
        labels = self.full_labels
        train_mask = self.full_train_mask

        self.optimizer.zero_grad()
        output = self.model(features, adj)
        loss = self.loss_fnc(output[train_mask], labels[train_mask])

        loss.backward()
        self.optimizer.step()

        self.stats.train_loss.append(loss.item())
        # train_score = self.calc_score(output[train_mask], labels[train_mask])
        # self.stats.train_scores.append(train_score)

        self.train_output = output
        # self.train_loss = loss.item()

        # print(end_time-start_time)

    @torch.no_grad()
    def validation(self, epoch):

        if epoch > 0 and epoch % self.config.val_step != 0:
            return True

        if self.config.cpu_val:
            model = copy.deepcopy(self.model).cpu()
        else:
            model = self.model
        
        model.eval()
        
        if self.train_output is None or self.config.dropout > 0:
            val_output = model(self.full_features, self.full_adj)
        else:
            val_output = self.train_output

        val_loss = self.loss_fnc(val_output[self.full_val_mask], self.full_labels[self.full_val_mask])
        val_score = self.calc_score(val_output[self.full_val_mask], self.full_labels[self.full_val_mask])

        if val_score > self.stats.best_val_score:
            self.stats.best_val_epoch = epoch
            self.stats.best_val_loss = val_loss.item()
            self.stats.best_val_score = val_score
            self.stats.best_model = copy.deepcopy(model)

        self.stats.val_loss.append(val_loss.item())
        self.stats.val_scores.append(val_score)

        # If train doesn't provide these (useful for other classes)
        if len(self.stats.train_loss) < epoch+1:
            train_loss = self.loss_fnc(val_output[self.full_train_mask], self.full_labels[self.full_train_mask])
            self.stats.train_loss.append(train_loss.item())

        if len(self.stats.train_scores) < epoch+1:
            train_score = self.calc_score(val_output[self.full_train_mask], self.full_labels[self.full_train_mask])
            self.stats.train_scores.append(train_score)

        return self.patience()

        # if len(self.stats.val_scores) > self.config.val_patience and \
        #         np.max(self.stats.val_scores[-1*self.config.val_patience:]) < self.stats.best_val_score:
        #     print('Run out of patience!')
        #     return False
        
        # return True

        # test_score  = self.calc_f1(val_output[self.full_test_mask], self.full_labels[self.full_test_mask])
        # print(f'#{epoch} '
        #         f'Loss: {self.stats.train_loss[-1]:.3f}, '
        #         # f'Train Score: {self.stats.train_scores[-1]*100:.2f}, '
        #         f'Val Score: {val_score*100:.2f}, '
        #         f'Test Score: {test_score*100:.2f}'
        #         )

    @torch.no_grad()
    def inference(self):
        self.stats.best_model.eval()
        test_preds = self.stats.best_model(self.full_features, self.full_adj)[self.full_test_mask]
        test_labels = self.full_labels[self.full_test_mask]
        test_score = self.calc_score(test_preds, test_labels)

        self.stats.test_score = test_score
