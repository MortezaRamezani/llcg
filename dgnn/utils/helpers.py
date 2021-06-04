import torch

def rank2dev(rank, num_gpus):
    if num_gpus == 0:
        device = torch.device('cpu')
    else:
        dev_id = rank % num_gpus
        device = torch.device('cuda:{}'.format(dev_id))
    return device