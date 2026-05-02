import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """Two-layer GraphSAGE for node-level prediction.

    References
    ----------
    .. [Hamilton2017] Hamilton, W. L., Ying, R., & Leskovec, J. (2017).
           "Inductive Representation Learning on Large Graphs." NeurIPS.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        aggr: str = "mean",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr=aggr)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
