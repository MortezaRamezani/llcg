import torch
import torch.distributed as dist
import torch_sparse

from ..utils import dist_sum, dist_spmm

class distgconv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, adjs, rank, world_size):

        ctx.save_for_backward(inputs, weight)
        ctx.adjs = adjs
        ctx.rank = rank
        ctx.world_size = world_size

        # T = AH
        T = dist_spmm(adjs, inputs, rank, world_size)
        # Z = TW
        Z = torch.mm(T, weight)

        return Z

    @staticmethod
    def backward(ctx, grad_output):
        
        inputs, weight = ctx.saved_tensors
        adjs = ctx.adjs
        rank = ctx.rank
        world_size = ctx.world_size

        ag = dist_spmm(adjs, grad_output, rank , world_size)

        grad_input = torch.mm(ag, weight.t())

        # grad_weight = dist_sum(ag, inputs.t(), rank, world_size)

        grad_weight = torch.mm(inputs.t(), ag)

        grad_weight = dist_sum(grad_weight)

        return grad_input, grad_weight, None, None, None


class DistGConv(torch.nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 rank=0,
                 ):       
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        self.fn = distgconv.apply
        self.weight = torch.nn.Parameter(torch.rand(input_dim, output_dim))


    def forward(self, x, adj):
        world_size = dist.get_world_size()
        x = self.fn(x, self.weight, adj, self.rank, world_size)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "[{}] ({}->{})".format(
            self.rank,
            self.input_dim,
            self.output_dim)
            