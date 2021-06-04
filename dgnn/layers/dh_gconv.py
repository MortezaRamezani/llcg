import torch
import torch.distributed as dist
import torch_sparse

from ..utils import dist_sum, dist_spmm


class dhgconv(torch.autograd.Function):

    fw_hist_t = []
    bw_hist_ag = []
    bw_hist_gw = []

    @staticmethod
    def forward(ctx, inputs, weight, adjs, rank, world_size, layerid, num_layers, use_hist=False):

        ctx.save_for_backward(inputs, weight)
        ctx.adjs = adjs
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.use_hist = use_hist
        ctx.layerid = layerid
        ctx.num_layers = num_layers

        if not use_hist:
            T = dist_spmm(adjs, inputs, rank, world_size)
            tmp_hist_t = T - adjs[rank].spmm(inputs)
            if len(dhgconv.fw_hist_t) < layerid + 1:
                dhgconv.fw_hist_t.append(tmp_hist_t)
            else:
                dhgconv.fw_hist_t[layerid] = tmp_hist_t
        else:
            # print('LayLay', layerid, len(dhgconv.fw_hist_t))
            T = adjs[rank].spmm(inputs) + dhgconv.fw_hist_t[layerid]
        
        # Z = TW
        Z = torch.mm(T, weight)

        return Z

    @staticmethod
    def backward(ctx, grad_output):
        
        inputs, weight = ctx.saved_tensors
        adjs = ctx.adjs
        rank = ctx.rank
        world_size = ctx.world_size
        use_hist = ctx.use_hist
        layerid = ctx.layerid
        num_layers = ctx.num_layers

        new_lid = num_layers - (layerid + 1)

        if not ctx.use_hist:
            ag = dist_spmm(adjs, grad_output, rank , world_size)
            tmp_hist_ag = ag  - adjs[rank].spmm(grad_output)
            if len(dhgconv.bw_hist_ag) < new_lid + 1:
                dhgconv.bw_hist_ag.append(tmp_hist_ag)
            else:
                dhgconv.bw_hist_ag[new_lid] = tmp_hist_ag
        else:
            ag = adjs[rank].spmm(grad_output) + dhgconv.bw_hist_ag[new_lid]

        grad_input = torch.mm(ag, weight.t())

        grad_weight = torch.mm(inputs.t(), ag)
        
        if not ctx.use_hist:
            tmp_gw = grad_weight
            grad_weight = dist_sum(grad_weight)
            tmp_hist_gw = grad_weight - tmp_gw

            if len(dhgconv.bw_hist_gw) < new_lid + 1:
                dhgconv.bw_hist_gw.append(tmp_hist_gw)
            else:
                dhgconv.bw_hist_gw[new_lid] = tmp_hist_gw
        else:
            grad_weight = grad_weight + dhgconv.bw_hist_gw[new_lid]


        return grad_input, grad_weight, None, None, None, None, None, None


# https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
# https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca
# https://pytorch.org/docs/stable/notes/extending.html#extending-torch-autograd
# https://www.kaggle.com/sironghuang/understanding-pytorch-hooks

class DHGConv(torch.nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 rank=0,
                 layerid=0,
                 num_layers=0,
                 ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.layerid = layerid
        self.num_layers = num_layers

        self.fn = dhgconv.apply
        self.weight = torch.nn.Parameter(torch.rand(input_dim, output_dim))


    def forward(self, x, adj, use_hist):
        world_size = dist.get_world_size()
        x = self.fn(x, self.weight, adj, self.rank, world_size, self.layerid, self.num_layers, use_hist)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.rank,
            self.input_dim,
            self.output_dim)
