import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    """Two-layer Graph Convolutional Network.

    Parameters
    ----------
    in_channels : int
        The number of input features.
    hidden_channels : int
        The number of hidden features.
    out_channels : int
        The number of output features.

    References
    ----------
    .. [Kipf2017] Kipf, T. N., & Welling, M. (2017).
           "Semi-Supervised Classification with Graph Convolutional Networks."
           arXiv:1609.02907.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        return self.conv2(x, edge_index)
