import torch
import torch.distributed as dist

from torch_sparse.matmul import matmul
from torch_sparse import spmm


def dist_sum(grad_weight):
    dist.all_reduce(grad_weight, op=dist.ReduceOp.SUM)
    # print('gw', grad_weight.numel() * grad_weight.element_size())
    return grad_weight


def dist_spmm(adjs, inputs, rank, world_size):
    device = adjs[0].device()

    t_buf = torch.FloatTensor(adjs[0].size(
        0), inputs.size(1)).fill_(0).to(device)

    # input_buf = torch.FloatTensor(adjs[0].size(0), inputs.size(1)).fill_(0).to(device)

    for i in range(world_size):
        if i == rank:
            input_buf = inputs.clone()
        else:
            # other parts maybe differnt sizes, otherwise broadcast freezes
            input_buf = torch.FloatTensor(adjs[i].size(
                1), inputs.size(1)).fill_(0).to(device)

        dist.broadcast(input_buf, src=i)
        # print('ib', input_buf.numel()*input_buf.element_size() )

        # buggy with CPU and GLOO
        t_buf += adjs[i].spmm(input_buf)
        # t_buf += matmul(adjs[i], input_buf)

        # new_adj = adjs[i].to_torch_sparse_coo_tensor()
        # t_buf += torch.spmm(new_adj, input_buf)

        # new_adj = adjs[i].to_torch_sparse_coo_tensor().coalesce()
        # t_buf += spmm(new_adj.indices(), new_adj.values(),
        #               new_adj.size(0), new_adj.size(1), input_buf)

        # Slow
        # index = torch.stack([adjs[i].storage.row(), adjs[i].storage.col()], dim=0)
        # value = adjs[i].storage.value()
        # t_buf += spmm(index, value, adjs[i].size(0), adjs[i].size(1), input_buf)

    return t_buf
