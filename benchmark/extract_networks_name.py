import requests
import json
from tqdm import tqdm

all_datasets = requests.get("https://networks.skewed.de/api/nets").json()

undirected_unipartite_nets = {}

progress_bar = tqdm(all_datasets)
for dataset in progress_bar:
    networks_url = f"https://networks.skewed.de/api/net/{dataset}"
    response = requests.get(networks_url).json()
    nets_list = response['nets']

    if len(nets_list)==1:
        is_direct = response['analyses']['is_directed']
        is_bipartite = response['analyses']['is_bipartite']
        if not is_direct and not is_bipartite:
            undirected_unipartite_nets[dataset] = [nets_list[0]]
        progress_bar.set_description(f"Processing {dataset} -  {nets_list[0]}")

    elif len(nets_list)>1:
        nets_list_undirected_unipartite = []
        for net in nets_list:
            is_direct = response['analyses'][net]['is_directed']
            is_bipartite = response['analyses'][net]['is_bipartite']
            if not is_direct and not is_bipartite:
                nets_list_undirected_unipartite.append(net)
        undirected_unipartite_nets[dataset] = nets_list_undirected_unipartite
        progress_bar.set_description(f"Processing {dataset} -  {nets_list_undirected_unipartite}")

with open('undirected_unipartite_nets.json', 'w') as f:
    json.dump(undirected_unipartite_nets, f)