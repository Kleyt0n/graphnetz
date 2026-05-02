import torch
from torch_geometric.data import Data
from torch_geometric.nn import DeepGraphInfomax, GCNConv


def _corruption(x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[torch.randperm(x.size(0))], edge_index


class _Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.prelu = torch.nn.PReLU(hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.conv(x, edge_index))


class DGI(torch.nn.Module):
    """Deep Graph Infomax for unsupervised node representation learning.

    References
    ----------
    .. [1] Veličković, P. et al. (2019). "Deep Graph Infomax." ICLR.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 512) -> None:
        super().__init__()
        self.model = DeepGraphInfomax(
            hidden_channels=hidden_channels,
            encoder=_Encoder(in_channels, hidden_channels),
            summary=lambda z, *_: torch.sigmoid(z.mean(dim=0)),
            corruption=_corruption,
        )

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(data.x, data.edge_index)

    def loss(self, pos_z: torch.Tensor, neg_z: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        return self.model.loss(pos_z, neg_z, summary)
