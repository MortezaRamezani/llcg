import os
import torch

from torch_geometric.datasets import *
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

from sklearn import preprocessing
import numpy as np

from .transforms import *

# https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html

class Dataset():
    """[summary]
    """
    # def __init__(self, dataset_name, split=None):
    #     # __new__ is called before __init__
    #     # hence it can returns another class object
     
    def __new__(cls, dataset_name, split=None):
        default_dir = os.path.join(os.path.expanduser('~'), '.gnn')
        dataset_dir = os.environ.get('GNN_DATASET_DIR', default_dir)

        # transform = T.Compose([T.NormalizeFeatures(), T.ToSparseTensor()])
        transform = T.Compose([T.ToSparseTensor()])

        # support shortened version of OGB dataset
        if dataset_name in ['arxiv', 'proteins', 'mag', 'products', 'papers100M']:
            dataset_name = 'ogbn-' + dataset_name

        if dataset_name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=dataset_dir, name=dataset_name, split='full', pre_transform=transform)
        elif dataset_name == 'reddit':
            dataset = Reddit(root=dataset_dir+'/reddit/', pre_transform=transform)
            setattr(dataset, 'name', 'reddit')
        elif dataset_name == 'yelp':
            dataset = Yelp(root=dataset_dir+'/yelp/', pre_transform=transform)
            setattr(dataset, 'name', 'yelp')
        elif dataset_name == 'flickr':
            dataset = Flickr(root=dataset_dir+'/flickr/' ,pre_transform=transform)
            setattr(dataset, 'name', 'flickr')
        elif dataset_name.startswith('ogbn'):

            if dataset_name == 'ogbn-proteins':
                transform = transform = T.Compose([T.ToSparseTensor(), PrepareProteins()])
            elif dataset_name == 'ogbn-arxiv':
                transform = transform = T.Compose([T.ToSparseTensor(), PrepareArxiv()])
            elif dataset_name == 'ogbn-products':
                transform = transform = T.Compose([T.ToSparseTensor(), PrepareProducts()])
            elif dataset_name == 'ogbn-papers100M':
                transform = transform = T.Compose([T.ToSparseTensor(), PreparePapers100M()])

            dataset = PygNodePropPredDataset(dataset_name, dataset_dir, pre_transform=transform)

            splitted_idx = dataset.get_idx_split()
            data = dataset.data

            # Fix few things about ogbn-proteins meta_info
            if dataset_name == 'ogbn-proteins':
                dataset.slices['x'] = dataset.slices['y']
                dataset.__num_classes__ = 112

            # Add split info to Data object
            for split in ['train', 'val', 'test']:
                mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                if split == 'val':
                    mask[splitted_idx['valid']] = True
                else:
                    mask[splitted_idx[split]] = True
                data[f'{split}_mask'] = mask
                dataset.slices[f'{split}_mask'] = dataset.slices['x']
            
            # data['val_mask'] = data['valid_mask']
            # dataset.slices['val_mask'] = dataset.slices['x']

        else:
            print('dataset {} is not supported!'.format(dataset_name))
            raise NotImplementedError

        return dataset