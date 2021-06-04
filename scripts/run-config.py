import os
import sys
import pdb
import argparse
import commentjson

sys.path.append('..')
sys.path.append('../dgnn/utils/cython/') # dirty hack to make up for relative import in pxd
from dgnn import data, utils, train

if os.environ['LOGNAME'] == 'mfr5226':
    os.environ['GNN_DATASET_DIR'] = '/export/local/mfr5226/datasets/pyg_dist/'

# base_dir = '../../../outputs/dist-gnn/721/'
# base_dir = '../../../outputs/dist-gnn/722/'
# base_dir = '../../../outputs/dist-gnn/723/'
base_dir = '../../../outputs/dist-gnn/724/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='cora')
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--np', type=int, default='8')
    parser.add_argument('--rep', type=int, default='1')
    parser.add_argument('--strat', action='store_true', default=False)
    parser.add_argument('--metis', action='store_true', default=False)
    parser.add_argument('--nosave', action='store_true', default=False)
    
    parser.add_argument('--bs', type=int, default='2048')
    parser.add_argument('--k', type=int, default='64')
    parser.add_argument('--s', type=int, default='1')

    args = parser.parse_args()


    if args.mode == 'full':
        trainer = train.Full
    elif args.mode == 'dgnnfull':
        trainer = train.serial.DistGNNFull
    elif args.mode == 'dgnnfullcor':
        trainer = train.serial.DistGNNFullCorr
    elif args.mode == 'sampling':
        trainer = train.Sampling
    elif args.mode == 'dgnn':
        trainer = train.serial.DistGNN
    elif args.mode == 'dgnncor':
        trainer = train.serial.DistGNNCorr
    elif args.mode == 'dgnnstale':
        trainer = train.serial.DistGNNStale
    elif args.mode == 'd2gnn':
        trainer = train.dist.DistGNN
    elif args.mode == 'd2gnncor':
        trainer = train.dist.DistGNNCorrection
    elif args.mode == 'dgl':
        trainer = train.dist.DistDGL
    else:
        raise NotImplementedError

    global_config = {
        'num_workers': args.np,
        'part_method': 'random' if not args.metis else 'metis',
        # 'part_method': 'metis',
        # 'part_method': 'overhead', 'part_args': 10,
        'weight_avg': True,
        'server_update': 1,
    }


    with open(f'./configs/{args.config}.json', 'r') as config_file:
        local_config = commentjson.load(config_file)
    
    dataset_name = local_config['dataset']
    dataset = data.Dataset(dataset_name)
    print('Done loading dataset...', dataset)
    out_dir = f'{base_dir}/{dataset_name}/'
    # run_name = trainer.__name__.lower()
    run_name = trainer.__module__.split('train.')[-1]
    tmp_global = {
        'dataset': dataset_name,
        'run_name': run_name,
        'output_dir': out_dir,
    }

    # import pdb; pdb.set_trace()

    if args.mode =="d2gnncor":
        tmp_global['server_opt_sync'] = True

    tmp_global['server_minibatch_size'] = args.bs
    tmp_global['local_updates'] = args.k
    tmp_global['server_updates'] = args.s

    global_config.update(local_config)
    global_config.update(tmp_global)

    
    print(trainer.__name__.lower(), dataset_name, global_config['num_workers'])
    print(global_config)

    train_config = utils.Config(global_config)

    for i in range(args.rep):
        exp = trainer(train_config, dataset)
        if i == 0:
            print(exp.model)
            
        print(f'Run #{i}...')
        exp.run()
        # import pdb; pdb.set_trace()
        if not args.nosave:
            exp.save()


# CUDA_VISIBLE_DEVICES=0,1,2 python run-config.py --config test --mode d2gnn --np 4 --nosave 2>/dev/null