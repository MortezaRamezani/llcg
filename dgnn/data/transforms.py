import torch

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul, coalesce
from torch_scatter import scatter_add


def row_norm(adj):
    if isinstance(adj, SparseTensor):
        # Add self loop
        adj_t = fill_diag(adj, 1)
        deg = sum(adj_t, dim=1)
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv.view(-1, 1))
        return adj_t


def col_norm(adj):
    if isinstance(adj, SparseTensor):
        # Add self loop
        adj_t = fill_diag(adj, 1)
        deg = sum(adj_t, dim=0)
        deg_inv = 1. / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv.view(-1, 1))
        return adj_t


def sym_norm(adj):
    if isinstance(adj, SparseTensor):
        adj_t = gcn_norm(adj) 
        return adj_t


class PrepareArxiv(object):
    """ Transformation for Arxiv for faster loading"""

    def __call__(self, data):
        data.adj_t = data.adj_t.to_symmetric()
        data.y = data.y.squeeze()
        del data.node_year
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class PrepareProducts(object):
    """ Transformation for Products for faster loading"""

    def __call__(self, data):
        # data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
        data.y = data.y.squeeze()
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class PrepareProteins(object):
    """ Prepare features and adjacency for Proteins """
        
    def __call__(self, data):

        ##! preprocessing from DeepGCN
        # le = preprocessing.LabelEncoder()
        # all_species = data.node_species # if I use dataset[0] here, dataset won't get updated!
        # species_unique = torch.unique(all_species)
        # max_no = species_unique.max()
        # le.fit(species_unique % max_no)
        # species = le.transform(all_species.squeeze() % max_no)
        # species = np.expand_dims(species, axis=1)
        # enc = preprocessing.OneHotEncoder()
        # enc.fit(species)
        # one_hot_encoding = enc.transform(species).toarray()
        # data.x = torch.from_numpy(one_hot_encoding).type(torch.FloatTensor)

        data.x = data.adj_t.mean(dim=1)
        data.adj_t.set_value_(None)
        data.y = data.y.to(torch.float)
        data.y = data.y.squeeze()

        # save space?
        del data.node_species

        return data
        

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class PreparePapers100M(object):
    """ Transformation for Papers100M for faster loading"""

    def __call__(self, data):
        data.adj_t = data.adj_t.to_symmetric()
        data.y = data.y.squeeze()
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)