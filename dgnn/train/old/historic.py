
import os
import copy

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from ..utils import Stats
from ..models import model_selector
from ..data import partition as P
from ..utils import helpers as H

from . import Distributed

class Historic(Distributed):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, rank, *args, **kwargs):
    
        # Load the rank-th partition, to rank-th device both adj and features
        adj, features, labels, train_mask, val_mask, test_mask = P.load_partitions(
            self.dataset_dir, rank)

        # Init the local model,
        model = copy.deepcopy(self.model)
        model.update_rank(rank)
        device = H.rank2dev(rank, self.num_gpus)

        print(device, flush=True)

        model = model.to(device)
        adj = [a.to(device) for a in adj]
        features = features.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        labels = labels.to(device)

        loss_fnc = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        best_model = copy.deepcopy(model)
        best_val_score = 0

        for epoch in range(self.config.num_epochs):
            # train
            model.train()
            optimizer.zero_grad()

            use_hist = None

            if self.config.hist_period > 1:
                use_hist = True

                if epoch == self.config.num_epochs - 1:
                        use_hist = False
                
                if not self.config.hist_exp:
                    if epoch % self.config.hist_period == 0:
                        use_hist = False
                else:
                    exp_pow = int(epoch / self.config.hist_period)
                    if epoch % pow(2, exp_pow) == 0:
                        use_hist = False

            output = model(features, adj, use_hist)
            loss = loss_fnc(output[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

            # Distributed Validation
            val_pred = output[val_mask].argmax(dim=1)
            val_acc = torch.stack(
                (val_pred.eq(labels[val_mask]).sum(), val_mask.sum()))

            all_val_acc = [torch.ones_like(val_acc) for _ in range(self.config.num_procs)]
            dist.all_gather(all_val_acc, val_acc)

            tmp_score = torch.stack(all_val_acc, dim=0).sum(dim=0)
            val_score = (tmp_score[0]/tmp_score[1]).item()

            if val_score > best_val_score:
                best_val_score = val_score
                best_model = copy.deepcopy(model)

            # End of Epoch
            if rank == 0:
                self.stats.train_loss.append(loss.item())
                self.stats.val_scores.append(val_score)

                print(f'Epoch  #{epoch}:',
                      f'train loss {loss.item():.3f}',
                      f'val accuracy {val_score*100:.2f}%',
                      '*' if not use_hist else '',
                      flush=True)

        # Testing
        best_model.eval()
        test_output = best_model(features, adj)

        test_pred = test_output[test_mask].argmax(dim=1)
        test_acc = torch.stack(
            (test_pred.eq(labels[test_mask]).sum(), test_mask.sum()))

        all_test_acc = [torch.ones_like(test_acc) for _ in range(self.config.num_procs)]
        dist.all_gather(all_test_acc, test_acc)

        if rank == 0:
            tmp_score = torch.stack(all_test_acc, dim=0).sum(dim=0)
            test_score = (tmp_score[0]/tmp_score[1]).item()

            self.stats.test_score = test_score

            print(f'Best model test score is: {test_score*100:.2f}%',
                  flush=True)

            self.save()