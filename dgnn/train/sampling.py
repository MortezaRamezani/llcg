from .base import Base
from .full import Full

from ..data import samplers
from ..data.transforms import row_norm

class Sampling(Full, Base):
    
    def __init__(self, config, dataset):

        # if full inference this, else Base init
        Full.__init__(self, config, dataset)
        # else:
        # Base.__init__(self, config, dataset)

        # Do it again
        self.full_adj = self.dataset[0].adj_t
        
        # if self.dataset.name.startswith('ogbn'):
        #     self.full_adj = self.dataset[0].adj_t.to_symmetric()

        if config.sampler == 'subgraph':
            self.train_loader = samplers.SubGraphSampler(self.full_adj,
                                                        self.config.minibatch_size,
                                                        num_workers=self.config.num_samplers,
                                                        num_batches=self.config.local_updates,
                                                        num_layers=self.config.num_layers,
                                                        persistent_workers=True,
                                                        )
        elif config.sampler == 'neighbor':
            self.train_loader = samplers.NeighborSampler(self.full_adj,
                                                        self.config.minibatch_size,
                                                        num_workers=self.config.num_samplers,
                                                        num_batches=self.config.local_updates, 
                                                        num_layers=self.config.num_layers,
                                                        num_neighbors=self.config.num_neighbors,
                                                        persistent_workers=True,
                                                        )
            # use row_norm for full inference
            self.full_adj = row_norm(self.full_adj).to(self.full_device)

        print(f'K={len(self.train_loader)}')

    def train(self, epoch):
        self.model.train()
        for input_nid, nodeblocks, output_nid in self.train_loader:

            # do this to train sampling with MLP
            #input_nid = output_nid
            
            # import pdb; pdb.set_trace()
            nodeblocks.to(self.device)
            features = self.full_features[input_nid]
            labels = self.full_labels[output_nid]
            train_mask = self.full_train_mask[output_nid]

            if self.config.cpu_val:
                features = features.to(self.device)
                labels = labels.to(self.device)
                train_mask = train_mask.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(features, nodeblocks)
            loss = self.loss_fnc(output[train_mask], labels[train_mask])

            loss.backward()
            self.optimizer.step()

            train_score = self.calc_score(output[train_mask], labels[train_mask])
            self.stats.train_loss.append(loss.item())
            self.stats.train_scores.append(train_score)

    # Sampling validation
    # @torch.no_grad()
    # def validation(self, epoch):
    #     raise NotImplementedError

    # Sampling inference
    # @torch.no_grad()
    # def inference(self):
    #     if self.config.sampling_infe:
    #       Full....            
    #     raise NotImplementedError

