import os
import os.path as osp
import requests
import pandas as pd
import networkx as nx
import torch
import zipfile
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from typing import Optional, Callable, List

class Netz(InMemoryDataset):
    """Netzschleuder dataset.
    
    Parameters
    ----------
    root : str
        Root directory where the dataset should be saved.
    dataset_name : str
        Name of the dataset from Netzschleuder repository.
    network_name : str
        Name of the network from Netzschleuder repository.
    split : str
        Split to load.

    Examples
    --------
    >>> dataset = Netzschleuder(root='data', dataset_name='london', network_name='london', split='train')

    >>> print(dataset[0])
    Data(x=[488, 488], edge_index=[2, 1458], num_nodes=488)

    References
    ----------
    Tiago P. Peixoto. (2020). The Netzschleuder network catalogue and repository. Zenodo. https://doi.org/10.5281/zenodo.7839981

    Notes
    -----
    The dataset is downloaded from the Netzschleuder repository. Links to the repository are available at: https://networks.skewed.de/
    
    """
    def __init__(self, root: str, 
                 dataset_name: str = 'urban_streets',
                 network_name: str = 'brasilia',
                 transform: Optional[Callable] = None, 
                 pre_transform: Optional[Callable] = None,
                 ):
        self.dataset_name = dataset_name
        self.network_name = network_name

        self.network_url = f'https://networks.skewed.de/net/{self.dataset_name}/files/{self.network_name}.csv.zip'

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.dataset_name, self.network_name, 'raw')
    
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.dataset_name, self.network_name, 'processed')
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['data.csv.zip']

    @property
    def processed_file_names(self) -> List[str]:
        return "data.pt"
    
    def download(self):
        # Download the network files
        response = requests.get(self.network_url)
        response.raise_for_status()

        # Save to raw directory
        os.makedirs(self.raw_dir, exist_ok=True)
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(raw_path, "wb") as f:
            f.write(response.content)

    def process(self) -> None:
        raw_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        
        # Extract the zip file
        with zipfile.ZipFile(raw_path, "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
    
        # Clean and read the edges file
        edges_file = os.path.join(self.raw_dir, 'edges.csv')
        with open(edges_file, "r") as f:
            lines = f.readlines()
        # Clean header (remove comments and spaces)
        header = lines[0].lstrip("#").replace(" ", "").strip() + "\n"
        lines[0] = header
        with open(edges_file, "w") as f:
            f.writelines(lines)
    
        # Read edges into pandas
        edges = pd.read_csv(edges_file)
        
        # Create appropriate networkx graph
        G = nx.from_pandas_edgelist(
            edges, 
            source="source", 
            target="target",
        )
        
        # Convert to PyTorch Geometric Data object
        data = from_networkx(G)
        
        # Add node features (identity matrix)
        data.x = torch.eye(data.num_nodes) 
        
        # Apply transformations and save data
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.dataset_name}()'
    

def download_all_networks_netz(root: str, 
             dataset_name: str,
             transform: Optional[Callable] = None, 
             pre_transform: Optional[Callable] = None):
    """Download and process all networks in the dataset."""
    networks_url = f"https://networks.skewed.de/api/net/{dataset_name}"
    response = requests.get(networks_url).json()['nets']

    for network_name in response:
        Netz(root, dataset_name, network_name, transform, pre_transform)


