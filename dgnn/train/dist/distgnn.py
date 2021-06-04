import os
import copy
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from multiprocessing import Value, Array
from ctypes import c_bool
from tqdm import trange

from ..base import Base
from ...data import samplers, partition
from ...utils import helpers as H
from ...data.transforms import row_norm


class DistGNN(Base):

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.ctx = mp.get_context('spawn')
        self.params_queue = self.ctx.Queue(maxsize=self.config.world_size*2)
        self.workers_flag = [self.ctx.Event() for _ in range(self.config.world_size)]
        self.workers_proc = []

        self.workers_meta_queue = self.ctx.Queue(maxsize=self.config.world_size)
        self.workers_train_size = [0] * self.config.world_size
        self.workers_total_train = 0

        # global model always stays on CPU
        self.global_model = copy.deepcopy(self.model)

        # CPU val and inference
        # Keep a global model on CPU and use GPU with another model
        self.val_device = 'cpu' if self.config.cpu_val else 'cuda:2'

        # check if full inference is set
        self.full_adj = self.dataset[0].adj_t
        
        # if self.dataset.name.startswith('ogbn') and self.dataset.name != 'ogbn-proteins':
        #     self.full_adj = self.full_adj.to_symmetric()

        if config.sampler == "neighbor":
            self.full_adj = row_norm(self.full_adj)
        else:
            self.full_adj = gcn_norm(self.full_adj)

        self.full_adj = self.full_adj.to(self.val_device)
        self.full_features = self.dataset[0].x.to(self.val_device)
        self.full_labels = self.dataset[0].y.to(self.val_device)
        self.full_train_mask = self.dataset[0].train_mask.to(self.val_device)
        self.full_val_mask = self.dataset[0].val_mask.to(self.val_device)
        self.full_test_mask = self.dataset[0].test_mask.to(self.val_device)

        self.model.to(self.val_device)

        self.end_train = Value(c_bool, False, lock=False)

        self.comm_cost = [Array('i', [0]*self.config.num_epochs, lock=False) for _ in  range(self.config.world_size)]

    def run(self):

        # Init the workers thread, create queue for param sharing
        for rank in range(self.config.world_size):
            p = self.ctx.Process(target=self.workers,
                                 args=(rank,
                                       self.params_queue,
                                       self.workers_flag,
                                       self.config,
                                       self.dataset.raw_dir,
                                       self.global_model,
                                       self.loss_fnc,
                                       self.workers_meta_queue,
                                       self.end_train,
                                       self.comm_cost[rank]
                                       )
                                 )
            p.start()
            self.workers_proc.append(p)

        # Collect meta data from workers
        for rank in range(self.config.world_size):
            tmp_tr_size = self.workers_meta_queue.get()
            self.workers_train_size[tmp_tr_size[0]] = tmp_tr_size[1]
        print('Train nodes per workers:', self.workers_train_size)
        self.workers_total_train = np.sum(self.workers_train_size)

        self.tbar = trange(self.config.num_epochs, desc='Epochs')

        # for epoch in range(self.config.num_epochs):
        for epoch in self.tbar:

            start_time = time.perf_counter()
            self.train(epoch)
            end_train_time = time.perf_counter()

            # check if they are different device, copy it, otherwise just assign to speedup
            # Copy global model for inference
            self.model.load_state_dict(self.global_model.state_dict())

            # self.validation(epoch)
            if not self.validation(epoch):
                self.end_train.value = True
                # Release all workers
                for rank in range(self.config.world_size):
                    self.workers_flag[rank].set()
                break
            
            end_val_time = time.perf_counter()
            
            train_loss = self.stats.train_loss[-1] if len(self.stats.train_loss) > 0 else 0
            train_score = self.stats.train_scores[-1] if len(self.stats.train_scores) > 0 else 0

            self.tbar.set_postfix(loss=f'{train_loss:.4f}',
                                  score=f'{train_score*100:.2f}',
                                  val_loss=f'{self.stats.val_loss[-1]:.4f}',
                                  val_score=f'{self.stats.val_scores[-1]*100:.2f}',
                                  )

            self.stats.train_time.append(end_train_time-start_time)
            self.stats.val_time.append(end_val_time-end_train_time)

        # Wait for all workers to finish, not necessary!
        for p in self.workers_proc:
            p.join()

        self.inference()
        end_inf_time = time.perf_counter()
        self.stats.test_time = end_inf_time - end_val_time

        print(f'Test Score: {self.stats.test_score * 100:.2f}% '
              f'@ Epoch #{self.stats.best_val_epoch}, '
              f'Highest Val: {self.stats.best_val_score * 100:.2f}%')
        
        print(f'Total training time: {np.sum(self.stats.train_time):.3f} sec.')

        # Save comm cost to stats
        comm_cost_list = []
        for rank in range(self.config.world_size):
            comm_cost_list.append(self.comm_cost[rank][:])
        self.stats.comm_cost = np.sum(comm_cost_list, axis=0)

    def train(self, epoch):

        # Collect local models
        local_params = []
        for rank in range(self.config.world_size):
            tmp_params = self.params_queue.get()
            local_params.append(tmp_params)

        # Do Param Averaging
        self.server_average(local_params, epoch)

        # Release workers for next epoch
        for rank in range(self.config.world_size):
            self.workers_flag[rank].set()

    def server_average(self, local_params, epoch):

        # params = self.model.state_dict()
        params = self.global_model.state_dict()
        for k in params.keys():
            params[k] = torch.zeros_like(params[k], dtype=torch.float, device='cpu')

        for rank in range(self.config.world_size):
            for k in params:
                params[k] += torch.div(local_params[rank][k] *
                                       self.workers_train_size[rank], self.workers_total_train)

        # self.model.load_state_dict(params)
        self.global_model.load_state_dict(params)



    # worker training
    @staticmethod
    def workers(rank, params_queue, ready_flag, config, dataset_rawdir,
                global_model, loss_fnc, meta_queue, end_train, comm_cost):

        ready_flag = ready_flag[rank]

        dataset_dir = os.path.join(dataset_rawdir[:-3], config.partitioned_dir)
        adj, feat, lab, tr_mask, _, _ = partition.load_partitions(dataset_dir, rank)

        meta_queue.put((rank, tr_mask.count_nonzero()))

        if config.part_method == 'metis':
            adj = adj[rank]

        # if config.dataset in ['arxiv']:
        #     adj = adj.to_symmetric()

        device = H.rank2dev(rank, config.num_gpus)

        if config.sampler == 'subgraph':
            train_loader = samplers.SubGraphSampler(adj,
                                                    config.minibatch_size,
                                                    num_workers=config.num_samplers,
                                                    num_batches=config.local_updates,
                                                    num_layers=config.num_layers,
                                                    persistent_workers=True,
                                                    )
        elif config.sampler == 'neighbor':
            train_loader = samplers.NeighborSampler(adj,
                                                    config.minibatch_size,
                                                    num_workers=config.num_samplers,
                                                    num_batches=config.local_updates,
                                                    num_layers=config.num_layers,
                                                    num_neighbors=config.num_neighbors,
                                                    persistent_workers=True,
                                                    )

        model = copy.deepcopy(global_model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        model_size = 0
        params = model.state_dict()
        for k in params:
            model_size += params[k].element_size() * params[k].nelement()

        for epoch in range(config.num_epochs):

            if end_train.value:
                break

            if epoch > 0:
                # Sync with Param Server
                model.load_state_dict(global_model.state_dict())

            model.train()
            ready_flag.clear()

            # Train Locally for K iterations
            for input_nid, nodeblocks, output_nid in train_loader:
                nodeblocks.to(device)
                features = feat[input_nid].to(device)
                labels = lab[output_nid].to(device)
                train_mask = tr_mask[output_nid].to(device)

                optimizer.zero_grad()
                output = model(features, nodeblocks)
                loss = loss_fnc(output[train_mask], labels[train_mask])
                loss.backward()
                optimizer.step()

            # Move to CPU and put on the Queue
            params_dict = {}
            tmp_params = model.state_dict()
            for key in tmp_params:
                params_dict[key] = tmp_params[key].clone().cpu()
            
            params_queue.put(params_dict)

            # Cost of communication
            comm_cost[epoch] = model_size

            # Wait for server to continue with new global_model
            ready_flag.wait()


    @torch.no_grad()
    def validation(self, epoch):

        if epoch > 0 and epoch % self.config.val_step != 0:
            return True

        self.model.eval()
        if self.train_output is None or self.config.dropout > 0:
            val_output = self.model(self.full_features, self.full_adj)
        else:
            val_output = self.train_output

        val_loss = self.loss_fnc(val_output[self.full_val_mask], self.full_labels[self.full_val_mask])
        val_score = self.calc_score(val_output[self.full_val_mask], self.full_labels[self.full_val_mask])

        self.stats.val_loss.append(val_loss.item())
        self.stats.val_scores.append(val_score)

        if val_score > self.stats.best_val_score:
            self.stats.best_val_epoch = epoch
            self.stats.best_val_loss = val_loss.item()
            self.stats.best_val_score = val_score
            self.stats.best_model = copy.deepcopy(self.model)

        # If train doesn't provide these (useful for other classes)
        if len(self.stats.train_loss) < epoch + 1:
            train_loss = self.loss_fnc(val_output[self.full_train_mask], self.full_labels[self.full_train_mask])
            self.stats.train_loss.append(train_loss.item())

        if len(self.stats.train_scores) < epoch + 1:
            train_score = self.calc_score(val_output[self.full_train_mask], self.full_labels[self.full_train_mask])
            self.stats.train_scores.append(train_score)

        return self.patience()

    @torch.no_grad()
    def inference(self):
        self.stats.best_model.eval()
        test_preds = self.stats.best_model(self.full_features, self.full_adj)[self.full_test_mask]
        test_labels = self.full_labels[self.full_test_mask]
        test_score = self.calc_score(test_preds, test_labels)

        self.stats.test_score = test_score



# print(f'Epoch #{epoch}: '
#       f'train loss: {train_loss:.4f}, '
#       f'train f1: {train_score*100:.2f}, '
#       f'val loss {self.stats.val_loss[-1]:.4f}, '
#       f'val f1 {self.stats.val_scores[-1]*100:.2f}.'
#       )