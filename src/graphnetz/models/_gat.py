import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """Two-layer Graph Attention Network.

    References
    ----------
    .. [Velickovic2018] Veličković, P. et al. (2018). "Graph Attention Networks." ICLR.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.nn.functional.elu(self.conv1(x, edge_index))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
