from .gcn import GCN
from .res_gcn import ResGCN
from .custom import Custom
from .dist_gcn import DistGCN
from .dh_gcn import DHGCN
from .gat import GAT
from .appnp import APPNP

def model_selector(model_str):
     # Selecting Model
    if model_str == 'gcn':
        model = GCN
    elif model_str == 'appnp':
        model = APPNP
    elif model_str == 'gat':
        model = GAT
    elif model_str == 'resgcn':
        model = ResGCN
    elif model_str == 'custom':
        model = Custom
    elif model_str == 'distgcn':
        model = DistGCN
    elif model_str == 'dhgcn':
        model = DHGCN
    else:
        return NotImplementedError

    return model