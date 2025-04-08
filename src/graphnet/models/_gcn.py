
import torch
from torch_geometric.nn import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCN(torch.nn.Module):
    """A simple GCN model.

    Parameters
    ----------
    in_channels : int
        The number of input features.
    hidden_channels : int
        The number of hidden features.
    out_channels : int
        The number of output features.

    Attributes
    ----------
    conv1 : GCNConv
        The first GCN layer.
    conv2 : GCNConv
        The second GCN layer.

    References
    ----------
    .. [1] Kipf, T. N., & Welling, M. (2017). 
           "Semi-Supervised Classification with Graph Convolutional Networks." 
           arXiv preprint arXiv:1609.02907.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels).to(device)
        self.conv2 = GCNConv(hidden_channels, out_channels).to(device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x