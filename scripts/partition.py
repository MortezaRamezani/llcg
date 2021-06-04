

import os
import sys
import argparse

sys.path.append('..')
sys.path.append('../dgnn/utils/cython/')
from dgnn import data
import dgnn.data.partition as P

if os.environ['LOGNAME'] == 'mfr5226':
    os.environ['GNN_DATASET_DIR'] = '/export/local/mfr5226/datasets/pyg_dist/'
else:
    os.environ['GNN_DATASET_DIR'] = '/home/weilin/Downloads/GCN_datasets/'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--num-parts', type=int, default=4)
    parser.add_argument('--mode', type=str, default='random')
    parser.add_argument('--overhead', type=int, default=10)


    config = parser.parse_args()

    dataset = data.Dataset(config.dataset)
    if config.mode == 'random':
        P.random(dataset, num_parts=config.num_parts)
    elif config.mode == 'metis':
        P.metis(dataset, num_parts=config.num_parts)
    elif config.mode == 'overhead':
        P.overhead(dataset, num_parts=config.num_parts, overhead=config.overhead)