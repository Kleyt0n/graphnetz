import torch
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv


class GraphTransformer(torch.nn.Module):
    """Two-layer graph transformer based on TransformerConv.

    References
    ----------
    .. [Shi2021] Shi, Y. et al. (2021). "Masked Label Prediction: Unified Message
           Passing Model for Semi-Supervised Classification." IJCAI.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)
