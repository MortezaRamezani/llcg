# from .gconv import GConv
from .dist_gconv import DistGConv
from .dh_gconv import DHGConv
from .gconv import GConv
from .mlp import MLPLayer
from .sage_conv import SAGEConv
from .residual import ResidualLayer
from .gatconv import GATConv
from .appnpconv import APPNPConv

def layer_selector(layer_str):
     # Selecting the layer
    if layer_str == 'gconv':
        layer = GConv
    elif layer_str == 'gatconv':
        layer = GATConv
    elif layer_str == 'appnpconv':
        layer = APPNPConv
    elif layer_str == 'mlp':
        layer = MLPLayer
    elif layer_str == 'sageconv':
        layer = SAGEConv
    else:
        return NotImplementedError
    return layer