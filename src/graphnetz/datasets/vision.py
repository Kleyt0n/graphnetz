"""Geometry and vision datasets.

Coverage:
- Image-derived superpixel graphs: ``MNISTSuperpixels``, ``CIFAR10`` (GNN benchmark).
- Meshes / point clouds: PyG ``ModelNet`` (10/40 classes), ``ShapeNet`` part segmentation.
"""

from torch_geometric.datasets import GNNBenchmarkDataset, MNISTSuperpixels, ModelNet, ShapeNet


def mnist_superpixels(root: str, train: bool = True) -> MNISTSuperpixels:
    """MNIST images converted to 75-superpixel graphs."""
    return MNISTSuperpixels(root=root, train=train)


def cifar10_superpixels(root: str, split: str = "train") -> GNNBenchmarkDataset:
    """CIFAR10 superpixel graphs (GNN benchmark suite)."""
    return GNNBenchmarkDataset(root=root, name="CIFAR10", split=split)


def modelnet10(root: str, train: bool = True) -> ModelNet:
    """ModelNet10 3D shapes (10 classes)."""
    return ModelNet(root=root, name="10", train=train)


def modelnet40(root: str, train: bool = True) -> ModelNet:
    """ModelNet40 3D shapes (40 classes)."""
    return ModelNet(root=root, name="40", train=train)


def shapenet(root: str, categories: list[str] | None = None) -> ShapeNet:
    """ShapeNet point clouds with part-segmentation labels.

    Pass ``categories=['Chair']`` (etc.) to limit to a subset.
    """
    return ShapeNet(root=root, categories=categories)


__all__ = [
    "cifar10_superpixels",
    "mnist_superpixels",
    "modelnet10",
    "modelnet40",
    "shapenet",
]
