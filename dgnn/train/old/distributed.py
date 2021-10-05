
import os
import copy
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from ...utils import Stats
from ...models import model_selector, DistGCN
from ...layers import layer_selector
from ...data import partition as P
from ...utils import helpers as H


class Distributed(object):

    def __init__(self,
                 config,
                 dataset,
                 ):

        # self.dataset = dataset
        self.config = config
        self.stats = Stats(config)

        self.dataset_dir = os.path.join(
            dataset.raw_dir, '../parted-{}'.format(self.config.world_size))
        self.raw_dir = dataset.raw_dir
        
        self.activation = torch.nn.ReLU(True)
        
        # model = model_selector(self.config.model)
        model = DistGCN
        layer = layer_selector(self.config.layer)
        

        self.model = model(
            dataset.num_features,
            self.config.hidden_size,
            dataset.num_classes,
            self.config.num_layers,
            self.activation,
            # layer=layer,
            # input_norm=self.config.input_norm,
            # layer_norm=self.config.layer_norm,
            # arch=self.config.arch,
            # residual=self.config.residual,
            # dropout=self.config.dropout,
        )

        self.backend = 'nccl'
        self.num_gpus = torch.cuda.device_count()
        
        if self.config.cpu:
            self.num_gpus = 0
        
        if self.config.cpu or self.config.world_size > self.num_gpus:
            self.backend = 'gloo'


    def run(self):
        
        # each process gets different copy of stack, can't store anything in original class object
        # solution: https://stackoverflow.com/questions/19828612/python-multiprocessing-setting-class-attribute-value
        # use mp.Value for shared values

        mp.spawn(self.start,
                 args=(self,),
                 nprocs=self.config.world_size,
                 join=True)

        # processes = []
        # ctx = mp.get_context('spawn')
        
        # for rank in range(self.config.num_procs):
        #     p = ctx.Process(target=self.run, args=(rank, self,))
        #     p.start()
        #     processes.append(p)
        
        # for p in processes:
        #     p.join()

    def save(self, *args, **kwargs):
        self.stats.save()

    def prepare(self, rank):
        pass

    def train(self, rank, *args, **kwargs):
    
        # Load the rank-th partition, to rank-th device both adj and features
        dataset_dir = os.path.join(self.raw_dir[:-3], self.config.partitioned_dir)
        adj, features, labels, train_mask, val_mask, test_mask = P.load_partitions(
           dataset_dir , rank)

        # Init the local model,
        model = copy.deepcopy(self.model)
        model.update_rank(rank)
        device = H.rank2dev(rank, self.num_gpus)

        print(device, flush=True)

        model = model.to(device)
        # if rank == 0:
        #     print(adj)
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

            output = model(features, adj)
            loss = loss_fnc(output[train_mask], labels[train_mask])

            loss.backward()
            optimizer.step()

        #     # Distributed Validation
        #     val_pred = output[val_mask].argmax(dim=1)
        #     val_acc = torch.stack(
        #         (val_pred.eq(labels[val_mask]).sum(), val_mask.sum()))

        #     all_val_acc = [torch.ones_like(val_acc) for _ in range(self.config.world_size)]
        #     dist.all_gather(all_val_acc, val_acc)

        #     tmp_score = torch.stack(all_val_acc, dim=0).sum(dim=0)
        #     val_score = (tmp_score[0]/tmp_score[1]).item()

        #     if val_score > best_val_score:
        #         best_val_score = val_score
        #         best_model = copy.deepcopy(model)

        #     # End of Epoch
        #     if rank == 0:
        #         self.stats.train_loss.append(loss.item())
        #         self.stats.val_scores.append(val_score)

        #         print(f'Epoch  #{epoch}:',
        #               f'train loss {loss.item():.3f}',
        #               f'val accuracy {val_score*100:.2f}%',
        #               flush=True)

        # # Testing
        # best_model.eval()
        # test_output = best_model(features, adj)

        # test_pred = test_output[test_mask].argmax(dim=1)
        # test_acc = torch.stack(
        #     (test_pred.eq(labels[test_mask]).sum(), test_mask.sum()))

        # all_test_acc = [torch.ones_like(test_acc) for _ in range(self.config.world_size)]
        # dist.all_gather(all_test_acc, test_acc)

        # if rank == 0:
        #     tmp_score = torch.stack(all_test_acc, dim=0).sum(dim=0)
        #     test_score = (tmp_score[0]/tmp_score[1]).item()

        #     self.stats.test_score.value = test_score

        #     print(f'Best model test score is: {test_score*100:.2f}%',
        #           flush=True)

        #     self.save()


    # multiple of this function is called
    @staticmethod
    def start(rank, cls):

        world_size = cls.config.world_size
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'

        # dist.init_process_group(backend='nccl', rank=rank,
        #                         world_size=world_size)

        dist.init_process_group(backend=cls.backend, rank=rank,
                                world_size=world_size)


        start_time = time.perf_counter()
        cls.train(rank)
        end_time = time.perf_counter()

        print('Total time: ', end_time - start_time)

        # # Load the rank-th partition, to rank-th device both adj and features
        # adj, features, labels, train_mask, val_mask, test_mask = P.load_partitions(
        #     cls.dataset_dir, rank)

        # # Init the local model,
        # model = copy.deepcopy(cls.model)
        # model.update_rank(rank)
        # device = H.rank2dev(rank, cls.config.num_procs)

        # print(device, flush=True)

        # model = model.to(device)
        # adj = [a.to(device) for a in adj]
        # features = features.to(device)
        # train_mask = train_mask.to(device)
        # val_mask = val_mask.to(device)
        # test_mask = test_mask.to(device)
        # labels = labels.to(device)

        # loss_fnc = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # best_model = copy.deepcopy(model)
        # best_val_score = 0

        # for epoch in range(cls.config.num_epochs):

        #     # train
        #     model.train()
        #     optimizer.zero_grad()

        #     use_hist = None
        #     if cls.config.hist_period > 1:
        #         use_hist = True
        #         if epoch % cls.config.hist_period == 0:
        #             use_hist = False

        #     output = model(features, adj, use_hist)

        #     loss = loss_fnc(output[train_mask], labels[train_mask])

        #     loss.backward()
        #     optimizer.step()

        #     # Distributed Validation
        #     # ! So unnecessary on all proc
        #     val_pred = output[val_mask].argmax(dim=1)
        #     val_acc = torch.stack(
        #         (val_pred.eq(labels[val_mask]).sum(), val_mask.sum()))

        #     all_val_acc = [torch.ones_like(val_acc) for _ in range(world_size)]
        #     dist.all_gather(all_val_acc, val_acc)

        #     tmp_score = torch.stack(all_val_acc, dim=0).sum(dim=0)
        #     val_score = (tmp_score[0]/tmp_score[1]).item()

        #     if val_score > best_val_score:
        #         best_val_score = val_score
        #         best_model = copy.deepcopy(model)

        #     # End of Epoch
        #     if rank == 0:
        #         cls.stats.train_loss.append(loss.item())
        #         cls.stats.val_scores.append(val_score)

        #         print(f'Epoch  #{epoch}:',
        #               f'train loss {loss.item():.3f}',
        #               f'val accuracy {val_score*100:.2f}%',
        #               flush=True)

        # # Testing
        # # best_model = copy.deepcopy(model)
        # best_model.eval()
        # test_output = best_model(features, adj)

        # test_pred = test_output[test_mask].argmax(dim=1)
        # test_acc = torch.stack(
        #     (test_pred.eq(labels[test_mask]).sum(), test_mask.sum()))

        # all_test_acc = [torch.ones_like(test_acc) for _ in range(world_size)]
        # dist.all_gather(all_test_acc, test_acc)

        # if rank == 0:
        #     tmp_score = torch.stack(all_test_acc, dim=0).sum(dim=0)
        #     test_score = (tmp_score[0]/tmp_score[1]).item()

        #     cls.stats.test_score = test_score

        #     print(f'Best model test score is: {test_score*100:.2f}%',
        #           flush=True)
            
        #     cls.save()
