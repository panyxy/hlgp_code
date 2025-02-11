import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self



class Dataset(Dataset):
    def __init__(self, node_coords, node_demands, capacities, remaining_capacities, via_depots):
        self.node_coords = node_coords
        self.node_demands = node_demands
        self.capacities = capacities
        self.remaining_capacities = remaining_capacities
        self.via_depots = via_depots

    def __len__(self):
        return len(self.node_coords)

    def __getitem__(self, item):
        node_coords = self.node_coords[item]
        node_demands = self.node_demands[item]
        capacities = self.capacities[item].to(torch.float)
        remaining_capacities = self.remaining_capacities[item]
        via_depots = self.via_depots[item].to(torch.long)
        distance_matrix = self.euclidean_distance(node_coords)

        item_dict = DotDict()
        item_dict.node_coords = node_coords
        item_dict.node_demands = node_demands
        item_dict.capacities = capacities
        item_dict.remaining_capacities = remaining_capacities
        item_dict.via_depots = via_depots
        item_dict.dist_matrices = distance_matrix

        return item_dict


    def euclidean_distance(self, node_coords):
        return (node_coords[:, None, :] - node_coords[None, :, :]).norm(p=2, dim=2)




def load_dataset(node_coords, node_demands, capacities, remaining_capacities, via_depots, batch_size, shuffle=False):
    dataset = Dataset(
        node_coords,
        node_demands,
        capacities,
        remaining_capacities,
        via_depots,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=shuffle,
        collate_fn=collate_func_with_sample
    )
    return dataloader


def collate_func_with_sample(dataset_items):
    graph_size = dataset_items[0].dist_matrices.shape[0]
    begin_idx = np.random.randint(0, graph_size - 3)

    new_dataset_items = []
    for data in dataset_items:
        new_data = {}
        for k, v in data.items():
            if k == 'dist_matrices':
                v_ = v[begin_idx:, begin_idx:]
            elif k == 'remaining_capacities':
                v_ = v[begin_idx]
            elif k == 'capacities':
                v_ = v
            else:
                v_ = v[begin_idx:, ...]

            new_data.update({k+'_s': v_})
        new_dataset_items.append({**data, **new_data})

    return default_collate(new_dataset_items)


