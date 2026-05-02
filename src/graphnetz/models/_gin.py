import torch
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool


def _mlp(in_dim: int, out_dim: int) -> Sequential:
    return Sequential(Linear(in_dim, out_dim), BatchNorm1d(out_dim), ReLU(), Linear(out_dim, out_dim), ReLU())


class GIN(torch.nn.Module):
    """Graph Isomorphism Network for graph-level prediction.

    References
    ----------
    .. [Xu2019] Xu, K. et al. (2019). "How Powerful are Graph Neural Networks?" ICLR.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 3) -> None:
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GINConv(_mlp(in_channels, hidden_channels), train_eps=True))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(_mlp(hidden_channels, hidden_channels), train_eps=True))
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_add_pool(x, batch)
        return self.classifier(x)
