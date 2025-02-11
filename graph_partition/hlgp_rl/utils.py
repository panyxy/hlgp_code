import copy
import os, sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from torch.nn import Module
import pickle
from torch.utils.data import Dataset, DataLoader

from insertion import random_insertion
from revision import reconnect
import random
import matplotlib.pyplot as plt
import math
from cvrp_env import OneShotCVRPEnv

from graph_partition.utils.lkh import lkh_solve

def get_cost_func(sol_coord, tour):
    return (sol_coord[:, 1:] - sol_coord[:, :-1]).norm(p=2, dim=2).sum(1)



def construct_local_dataset(instances, demands, indices, cost_list, n_graph_array, depot_coord, order=None, shifts=0, replace=False, n_neigh=2, min_max=False):
    batch_size, beam_size = n_graph_array.shape
    n_graphs, graph_size, _ = instances.size()
    device = instances.device

    assert n_graph_array.sum() == n_graphs
    assert batch_size == depot_coord.size(0)

    neigh_coord, neigh_demand, neigh_indice, neigh_depot, neigh_cost = [], [], [], [], []
    max_vehicle_list, mask_depot_list = [], []

    cum_n_graphs = 0
    for i in range(batch_size):
        for j in range(beam_size):
            subinst = instances[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]]
            subdem = demands[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]]
            subind = indices[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]]
            subcost = cost_list[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]]

            suborder = order[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]]
            first_index = torch.where(suborder == 0)[0][0].item()
            suborder = suborder.roll(dims=0, shifts=shifts - first_index)

            subinst = subinst.gather(dim=0, index=suborder[:, None, None].expand_as(subinst))
            subdem = subdem.gather(dim=0, index=suborder[:, None].expand_as(subdem))
            subind = subind.gather(dim=0, index=suborder[:, None].expand_as(subind))
            subcost = subcost.gather(dim=0, index=suborder)

            if replace:
                instances[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]] = subinst
                demands[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]] = subdem
                indices[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]] = subind
                cost_list[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]] = subcost
                order[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]] = suborder

            neigh_padding = n_neigh - int(subinst.size(0) % n_neigh)
            if neigh_padding < n_neigh:
                subinst = torch.cat((subinst, depot_coord[i:i+1][None].expand(neigh_padding, graph_size, 2)), dim=0)
                subdem = torch.cat((subdem, torch.zeros((neigh_padding, graph_size), device=device, dtype=torch.int)), dim=0)
                subind = torch.cat((subind, torch.zeros((neigh_padding, graph_size), device=device, dtype=torch.int)), dim=0)
                subcost = torch.cat((subcost, torch.zeros((neigh_padding, ), device=device, dtype=torch.float)), dim=0)

            neigh_coord.append(torch.cat([subinst[s_idx::n_neigh] for s_idx in range(n_neigh)], dim=1))
            neigh_demand.append(torch.cat([subdem[s_idx::n_neigh] for s_idx in range(n_neigh)], dim=1))
            neigh_indice.append(torch.cat([subind[s_idx::n_neigh] for s_idx in range(n_neigh)], dim=1))
            neigh_cost.append(torch.stack([subcost[s_idx::n_neigh] for s_idx in range(n_neigh)], dim=1))
            neigh_depot.append(depot_coord[i:i + 1].expand(int(subinst.size(0) // n_neigh), 2))

            max_vehicle = torch.full(size=(int(subinst.size(0) // n_neigh), ), device=device, dtype=torch.long, fill_value=n_neigh)
            max_vehicle[-1] = max_vehicle[-1] - neigh_padding if neigh_padding < n_neigh else max_vehicle[-1]
            mask_depot = ((neigh_coord[-1] - depot_coord[i:i+1]).norm(p=2, dim=2) >= 1e-6).to(torch.int)

            max_vehicle_list.append(max_vehicle)
            mask_depot_list.append(mask_depot)

            cum_n_graphs += n_graph_array[i, j]


    neigh_coord = torch.cat(neigh_coord, dim=0)
    neigh_demand = torch.cat(neigh_demand, dim=0)
    neigh_indice = torch.cat(neigh_indice, dim=0)
    neigh_cost = torch.cat(neigh_cost, dim=0)
    neigh_depot = torch.cat(neigh_depot, dim=0)[:, None]

    max_vehicle_list = torch.cat(max_vehicle_list, dim=0)
    mask_depot_list = torch.cat(mask_depot_list, dim=0)

    if min_max:
        return (neigh_depot, neigh_coord, neigh_demand), neigh_cost.max(dim=-1, keepdim=True), neigh_indice, max_vehicle_list, mask_depot_list
    else:
        return (neigh_depot, neigh_coord, neigh_demand), neigh_cost.sum(dim=-1, keepdim=True), neigh_indice, max_vehicle_list, mask_depot_list




def local_partition(instances, demands, indices, cost_list, n_graph_array, depot_coord, capacity,
                    loc_part_model, perm_model, args, direction='forward', shifts=0, beam_size=None,
                    sampling_type=None, use_mode='validation', n_neigh=2, perm_model_type='am_reviser'):

    perm_model = [perm_model] if not isinstance(perm_model, List) else perm_model
    assert n_graph_array.shape[1] == 1

    device = args.loc_part_device
    problem_size = int(n_neigh * instances.size(1))

    order = reorder_instances_by_endpoint(instances, n_graph_array, depot_coord, direction=direction)
    neigh_instances, neigh_cost, neigh_indices, neigh_max_vehicle, neigh_mask_depot = construct_local_dataset(
        instances, demands, indices, cost_list, n_graph_array, depot_coord, order=order, shifts=shifts, replace=True, n_neigh=n_neigh, min_max=args.min_max
    )
    neigh_graph_array = np.ceil(n_graph_array / n_neigh).astype(int)

    neigh_indices = torch.cat((torch.zeros(neigh_indices.size(0), 1, device=device, dtype=torch.long), neigh_indices), dim=1)
    env = OneShotCVRPEnv(args, neigh_instances[0].size(0), problem_size, beam_size, device)
    env.load_problems(neigh_instances + (capacity, ))

    reset_state = env.reset(max_vehicle=neigh_max_vehicle, mask_depot=neigh_mask_depot)
    heatmap = loc_part_model(reset_state)
    log_probs_list = env.step(heatmap, sampling_type=sampling_type)

    new_instances, new_demands, new_indices, new_ng_array = get_subinstances(
        env.depot_node_xy, env.depot_node_demand, env.selected_node_list, env.depot_idx_list, depot_node_indice=neigh_indices
    )
    new_instances, new_demands, new_indices, new_cost_list, new_cost = permute_partition(
        args, perm_model, new_instances, new_demands, new_indices, new_ng_array, perm_model_type=perm_model_type
    )

    ori_ng_array = neigh_max_vehicle.cpu().numpy()[:, None]
    assert neigh_graph_array.sum() == ori_ng_array.shape[0] == new_ng_array.shape[0]

    if use_mode == 'train_local':
        assert new_cost.size(1) > 1
        new_cost = new_cost.to(device)
        cost_diff = new_cost - neigh_cost

        obj_loss = ((cost_diff - cost_diff.mean(dim=1, keepdim=True)) * log_probs_list.sum(dim=2)).mean()
        entropy_loss = -torch.log(heatmap / heatmap.sum(dim=2, keepdim=True)).mean()
        loss = obj_loss + entropy_loss * args.loc_lambda


    if beam_size > 1:
        new_instances, new_demands, new_indices, new_cost_list, new_cost, new_ng_array = select_partition_from_sampling(
            new_instances, new_demands, new_indices, new_cost_list, new_cost, new_ng_array, topK=1
        )

    selected_instances, selected_demands, selected_indices, selected_cost, n_graph_array = select_partition_from_baseline(
        new_instances, new_demands, new_indices, new_cost, new_ng_array,
        instances, demands, indices, neigh_cost, ori_ng_array, neigh_instances, neigh_graph_array
    )

    if use_mode == 'train_local':
        return selected_instances, selected_demands, selected_indices, selected_cost, n_graph_array, loss
    elif use_mode == 'train_global':
        return selected_instances, selected_demands, selected_indices, selected_cost, n_graph_array
    elif use_mode == 'validation':
        return selected_instances, selected_demands, selected_indices, selected_cost, n_graph_array





def select_partition_from_sampling(new_instances, new_demands, new_indices, new_cost_list, new_cost, new_ng_array, topK=1):
    assert new_ng_array.shape[1] > 1
    new_cost_idx = new_cost.sort(dim=1, descending=False)[1]
    new_cost_idx = new_cost_idx[:, :topK].sort(dim=1, descending=False)[0]
    new_cost = new_cost.gather(dim=1, index=new_cost_idx)

    cum_num = 0
    selected_instances, selected_demands, selected_indices, selected_cost_list, selected_ng_array = [], [], [], [], []
    for i in range(new_ng_array.shape[0]):
        for j in range(new_ng_array.shape[1]):
            if j in new_cost_idx[i]:
                selected_instances.append(new_instances[cum_num: cum_num + new_ng_array[i, j]])
                selected_demands.append(new_demands[cum_num: cum_num + new_ng_array[i, j]])
                selected_indices.append(new_indices[cum_num: cum_num + new_ng_array[i, j]])
                selected_cost_list.append(new_cost_list[cum_num: cum_num + new_ng_array[i, j]])
                selected_ng_array.append(new_ng_array[i, j])

            cum_num += new_ng_array[i, j]


    new_instances = torch.cat(selected_instances, dim=0)
    new_demands = torch.cat(selected_demands, dim=0)
    new_indices = torch.cat(selected_indices, dim=0)
    new_cost_list = torch.cat(selected_cost_list, dim=0)
    new_ng_array = np.array(selected_ng_array).reshape(new_ng_array.shape[0], topK)

    return new_instances, new_demands, new_indices, new_cost_list, new_cost, new_ng_array




def select_partition_from_baseline(new_instances, new_demands, new_indices, new_cost, new_ng_array,
                                   instances, demands, indices, neigh_cost, ori_ng_array,
                                   neigh_instances, neigh_graph_array, max_graph_size=50):
    device = instances.device
    new_instances = new_instances.to(device)
    new_demands = new_demands.to(device)
    new_indices = new_indices.to(device)
    new_cost = new_cost.to(device)

    ori_sum = new_sum = 0
    selected_instances, selected_demands, selected_indices, selected_ng_array = [], [], [], []
    max_graph_size = max_graph_size
    for i in range(new_ng_array.shape[0]):
        for j in range(new_ng_array.shape[1]):
            if new_cost[i, j] >= neigh_cost[i, j] or ori_ng_array[i, j] < new_ng_array[i, j]:
                selected_instances.append(instances[ori_sum: ori_sum + ori_ng_array[i, j]])
                selected_demands.append(demands[ori_sum: ori_sum + ori_ng_array[i, j]])
                selected_indices.append(indices[ori_sum: ori_sum + ori_ng_array[i, j]])
                selected_ng_array.append(ori_ng_array[i, j])
            else:
                selected_instances.append(new_instances[new_sum: new_sum + new_ng_array[i, j]])
                selected_demands.append(new_demands[new_sum: new_sum + new_ng_array[i, j]])
                selected_indices.append(new_indices[new_sum: new_sum + new_ng_array[i, j]])
                selected_ng_array.append(new_ng_array[i, j])

            max_graph_size = max(max_graph_size, selected_instances[-1].size(1))
            ori_sum += ori_ng_array[i, j]
            new_sum += new_ng_array[i, j]

    assert len(selected_instances) == new_ng_array.shape[0] == ori_ng_array.shape[0]
    assert len(selected_ng_array) == neigh_graph_array.sum() == neigh_instances[0].size(0)
    assert neigh_graph_array.shape[1] == 1

    for i in range(len(selected_instances)):
        n_padding = max_graph_size - selected_instances[i].size(1)
        if n_padding > 0:
            sel_inst, sel_dem, sel_ind = [], [], []
            for j in range(selected_instances[i].size(0)):
                idx = torch.where(selected_demands[i][j] == 0)[0][0]
                sel_inst.append(torch.cat((selected_instances[i][j, :idx], neigh_instances[0][i].expand(n_padding, 2), selected_instances[i][j, idx:]), dim=0))
                sel_dem.append(torch.cat((selected_demands[i][j, :idx], torch.zeros(n_padding, device=device, dtype=torch.int), selected_demands[i][j, idx:]), dim=0))
                sel_ind.append(torch.cat((selected_indices[i][j, :idx], torch.zeros(n_padding, device=device, dtype=torch.long), selected_indices[i][j, idx:]), dim=0))
            selected_instances[i] = torch.stack(sel_inst, dim=0)
            selected_demands[i] = torch.stack(sel_dem, dim=0)
            selected_indices[i] = torch.stack(sel_ind, dim=0)

    selected_instances = torch.cat(selected_instances, dim=0)
    selected_demands = torch.cat(selected_demands, dim=0)
    selected_indices = torch.cat(selected_indices, dim=0)
    selected_cost = (selected_instances - selected_instances.roll(dims=1, shifts=1)).norm(p=2, dim=2).sum(dim=1)

    selected_ng_array = np.array(selected_ng_array)
    neigh_graph_array = neigh_graph_array.reshape((-1, ))
    cum_num = 0
    n_graph_array = []
    for i in range(len(neigh_graph_array)):
        n_graph_array.append(selected_ng_array[cum_num: cum_num+neigh_graph_array[i]].sum())
        cum_num += neigh_graph_array[i]
    n_graph_array = np.array(n_graph_array)[:, None]

    assert n_graph_array.sum() == selected_instances.size(0)
    return selected_instances, selected_demands, selected_indices, selected_cost, n_graph_array




def construct_partial_instances(subindices, cost_list, n_graph_array, depot_node_xy, depot_node_demand, depot_node_indice, capacity, cut_index, min_max=False):
    batch_size, beam_size = n_graph_array.shape
    _, graph_size, _ = depot_node_xy.size()
    assert depot_node_demand.size() == depot_node_indice.size() == (batch_size, graph_size)
    assert depot_node_xy.size(0) == batch_size
    assert cut_index.size() == (batch_size, beam_size, 2)
    assert beam_size == 1

    n_subgraphs, subgraph_size = subindices.size()
    assert cost_list.size(0) == n_subgraphs

    device = depot_node_xy.device
    subindices = subindices.to(device)
    cost_list = cost_list.to(device)

    base_costs = []
    node_indices = []
    cum_num = 0
    max_length = 0
    for i in range(batch_size):
        for j in range(beam_size):
            subind = subindices[cum_num: cum_num + n_graph_array[i, j]]
            subcost = cost_list[cum_num: cum_num + n_graph_array[i, j]]

            cut_idx0 = cut_index[i, j, 0]
            cut_idx1 = cut_index[i, j, 1] + n_graph_array[i, j] + 1 if cut_index[i, j, 1] < 0 else cut_index[i, j, 1] + 1
            assert 0 <= cut_idx0 < cut_idx1 <= n_graph_array[i, j]

            seg_subind = torch.reshape(subind[cut_idx0: cut_idx1], ((cut_idx1 - cut_idx0) * subgraph_size, ))
            seg_subind = torch.unique(seg_subind)
            node_indices.append(seg_subind)

            if min_max:
                seg_subcost = subcost[cut_idx0: cut_idx1].max()
            else:
                seg_subcost = subcost[cut_idx0: cut_idx1].sum()
            base_costs.append(seg_subcost)

            max_length = max(max_length, seg_subind.size(0))
            cum_num += n_graph_array[i, j]

    node_indices = [
        F.pad(subind, (0, max_length - subind.size(0)), mode='constant', value=0)
        for subind in node_indices
    ]
    node_indices = torch.stack(node_indices, dim=0)
    base_costs = torch.stack(base_costs, dim=0)[:, None]

    depot_coords = depot_node_xy[:, 0:1]
    node_coords = depot_node_xy.gather(dim=1, index=node_indices[:, :, None].repeat(1, 1, 2))
    node_demands = depot_node_demand.gather(dim=1, index=node_indices)
    mask_depot = (node_demands != 0).to(torch.float)

    depot_node_indice = depot_node_indice.gather(dim=1, index=node_indices)
    depot_node_indice = torch.cat((torch.zeros((batch_size, 1), device=device, dtype=torch.long), depot_node_indice), dim=1)

    instances = (depot_coords, node_coords, node_demands, int(capacity))
    return instances, depot_node_indice, mask_depot, base_costs


def select_partial_data(bkp_subindices, bkp_cost_list, bkp_n_graph_array, bkp_costs, new_subindices, new_cost_list, new_n_graph_array, new_costs, cut_index, env_depot_node_indice):

    def padding_func(tensor, max_length):
        if max_length == tensor.size(1): return tensor
        if tensor.size(0) == 0: return torch.zeros((0, max_length), device=tensor.device, dtype=tensor.dtype)

        pad_num = max_length - tensor.size(1)
        device = tensor.device
        tensor_ls = []
        for k in range(tensor.size(0)):
            pad_idx = torch.where(tensor[k] == 0)[0][0]
            tensor_ls.append(
                torch.cat((tensor[k][0:pad_idx],
                           torch.zeros((pad_num,), device=device, dtype=tensor.dtype),
                           tensor[k][pad_idx:]), dim=0)
            )
        return torch.stack(tensor_ls, dim=0)


    assert bkp_n_graph_array.sum() == bkp_subindices.size(0) == bkp_cost_list.size(0)
    assert new_n_graph_array.sum() == new_subindices.size(0) == new_cost_list.size(0)

    batch_size, beam_size, _ = cut_index.size()
    assert bkp_n_graph_array.shape == bkp_costs.size() == (batch_size, beam_size)
    assert new_n_graph_array.shape == new_costs.size() == (batch_size, beam_size)
    assert (env_depot_node_indice.size(0), 1) == (batch_size, beam_size)

    device = env_depot_node_indice.device
    new_subindices = new_subindices.to(device)
    new_cost_list = new_cost_list.to(device)
    new_costs = new_costs.to(device)

    sel_subindices, sel_cost_list, sel_ng_array = [], [], []
    bkp_cum_num = 0
    new_cum_cum = 0
    max_length = 0
    for i in range(batch_size):
        for j in range(beam_size):
            bkp_subind = bkp_subindices[bkp_cum_num: bkp_cum_num + bkp_n_graph_array[i, j]]
            bkp_subcost = bkp_cost_list[bkp_cum_num: bkp_cum_num + bkp_n_graph_array[i, j]]

            new_subind = new_subindices[new_cum_cum: new_cum_cum + new_n_graph_array[i, j]]
            new_subcost = new_cost_list[new_cum_cum: new_cum_cum + new_n_graph_array[i, j]]

            depot_node_indice = env_depot_node_indice[i:i+1].repeat(new_n_graph_array[i, j], 1).clone()
            new_subind = depot_node_indice.gather(index=new_subind, dim=1)

            cut_idx0 = cut_index[i, j, 0]
            cut_idx1 = cut_index[i, j, 1] + bkp_n_graph_array[i, j] + 1 if cut_index[i, j, 1] < 0 else cut_index[i, j, 1] + 1
            if new_costs[i, j] >= bkp_costs[i, j]:
                tmp_subind = bkp_subind[cut_idx0: cut_idx1]
                tmp_subcost = bkp_subcost[cut_idx0: cut_idx1]
                tmp_ng = (cut_idx1 - cut_idx0).item()
            else:
                tmp_subind = new_subind
                tmp_subcost = new_subcost
                tmp_ng = new_n_graph_array[i, j]
            tmp_max_length = max(tmp_subind.size(1), bkp_subind.size(1))

            sel_subindices.append(torch.cat(
                (padding_func(bkp_subind[:cut_idx0], tmp_max_length),
                 padding_func(tmp_subind, tmp_max_length),
                 padding_func(bkp_subind[cut_idx1:], tmp_max_length),
                 ), dim=0
            ))

            sel_cost_list.append(torch.cat((bkp_subcost[:cut_idx0], tmp_subcost, bkp_subcost[cut_idx1:]), dim=0))
            sel_ng_array.append(bkp_n_graph_array[i, j] - (cut_idx1 - cut_idx0).item() + tmp_ng)

            bkp_cum_num += bkp_n_graph_array[i, j]
            new_cum_cum += new_n_graph_array[i, j]
            max_length = max(max_length, sel_subindices[-1].size(1))


    for i in range(len(sel_subindices)):
        sel_subindices[i] = padding_func(sel_subindices[i], max_length)

    sel_subindices = torch.cat(sel_subindices, dim=0)
    sel_cost_list = torch.cat(sel_cost_list, dim=0)
    sel_ng_array = np.array(sel_ng_array).reshape((batch_size, beam_size))

    return sel_subindices, sel_cost_list, sel_ng_array


def random_cut_index(n_graph_array, device):
    batch_size, beam_size = n_graph_array.shape
    cut_index = []
    for i in range(batch_size):
        for j in range(beam_size):
            ng = n_graph_array[i, j]
            cut_index.append(sorted(np.random.choice(np.arange(ng), size=(2, ), replace=False)))
    return torch.tensor(cut_index, device=device).reshape((batch_size, beam_size, 2))

def subp_cut_index(n_graph_array, device, bkp_cut_index=None):
    batch_size, beam_size = n_graph_array.shape

    if bkp_cut_index == None:
        cut_index = [[1, -1] for _ in range(batch_size * beam_size)]
    else:
        cut_index = []
        for i in range(batch_size):
            for j in range(beam_size):
                ng = n_graph_array[i, j]
                bkp_idx0, bkp_idx1 = bkp_cut_index[i, j, 0], bkp_cut_index[i, j, 1]

                if bkp_idx0 + 1 < ng + bkp_idx1:
                    cut_index.append([bkp_idx0 + 1, bkp_idx1])
                else:
                    if ng + bkp_idx1 - 1 != 0:
                        cut_index.append([0, bkp_idx1 - 1])
                    else:
                        cut_index.append([0, -1])

    return torch.tensor(cut_index, device=device).reshape((batch_size, beam_size, 2))



def construct_tail_instances(subindices, cost_list, n_graph_array, depot_node_xy, depot_node_demand, depot_node_indice, capacity, n_remove=1, min_max=False):
    batch_size, beam_size = n_graph_array.shape
    n_subgraphs, subgraph_size = subindices.size()
    _, graph_size, _ = depot_node_xy.size()

    device = depot_node_xy.device
    subindices = subindices.to(device)
    cost_list = cost_list.to(device)

    assert n_graph_array.sum() == n_subgraphs == cost_list.size(0)
    assert beam_size == 1
    assert batch_size == depot_node_xy.size(0)
    assert depot_node_demand.size() == depot_node_indice.size() == (batch_size, graph_size)

    bkp_subind, bkp_subcost, bkp_ng_array = [], [], []
    rem_xy_ls, rem_demand_ls, rem_indice_ls, rem_cost_ls = [], [], [], []
    base_costs = []
    valid_instance = []
    node_indices = []
    cum_num = 0
    max_length = 0
    for i in range(batch_size):
        for j in range(beam_size):
            subind = subindices[cum_num: cum_num + n_graph_array[i, j]]
            subcost = cost_list[cum_num: cum_num + n_graph_array[i, j]]

            if n_graph_array[i, j] > n_remove:
                tail_subind = torch.reshape(subind[n_remove: ], ((n_graph_array[i, j] - n_remove) * subgraph_size, ))
                tail_subind = torch.unique(tail_subind)
                node_indices.append(tail_subind)

                if min_max:
                    tail_subcost = subcost[n_remove: ].max()
                else:
                    tail_subcost = subcost[n_remove: ].sum()
                base_costs.append(tail_subcost)

                valid_instance.append(True)
                max_length = max(max_length, tail_subind.size(0))

                bkp_subind.append(subind[n_remove:])
                bkp_subcost.append(subcost[n_remove:])
                bkp_ng_array.append(n_graph_array[i, j] - n_remove)
            else:
                valid_instance.append(False)
            cum_num += n_graph_array[i, j]
            rem_num = min(n_remove, n_graph_array[i, j])

            tmp_xy = depot_node_xy[i:i+1].repeat(rem_num, 1, 1).clone()
            rem_xy = tmp_xy.gather(dim=1, index=subind[:rem_num][:, :, None].repeat(1, 1, 2))
            rem_xy_ls.append(rem_xy)

            tmp_demand = depot_node_demand[i:i+1].repeat(rem_num, 1).clone()
            rem_demand = tmp_demand.gather(dim=1, index=subind[:rem_num])
            rem_demand_ls.append(rem_demand)

            tmp_indice = depot_node_indice[i:i+1].repeat(rem_num, 1).clone()
            rem_indice = tmp_indice.gather(dim=1, index=subind[:rem_num])
            rem_indice_ls.append(rem_indice)

            rem_cost = subcost[:rem_num]
            rem_cost_ls.append(rem_cost)


    valid_instance = np.array(valid_instance)
    if not (valid_instance == False).all():
        node_indices = [
            F.pad(subind, (0, max_length - subind.size(0)), mode='constant', value=0)
            for subind in node_indices
        ]
        node_indices = torch.stack(node_indices, dim=0)
        base_costs = torch.stack(base_costs, dim=0)[:, None]

        depot_node_xy = depot_node_xy[valid_instance]
        depot_node_demand = depot_node_demand[valid_instance]
        depot_node_indice = depot_node_indice[valid_instance]
        assert depot_node_xy.size(0) == depot_node_demand.size(0) == depot_node_indice.size(0) == node_indices.size(0)

        bkp_data = (
            torch.cat(bkp_subind, dim=0),
            torch.cat(bkp_subcost, dim=0),
            np.array(bkp_ng_array).reshape((len(bkp_ng_array), 1)),
            depot_node_xy.clone(),
            depot_node_demand.clone(),
            depot_node_indice.clone(),
            int(capacity),
        )
        rem_data = (
            rem_xy_ls,
            rem_demand_ls,
            rem_indice_ls,
            rem_cost_ls,
        )

        depot_coords = depot_node_xy[:, 0:1]
        node_coords = depot_node_xy.gather(dim=1, index=node_indices[:, :, None].repeat(1, 1, 2))
        node_demands = depot_node_demand.gather(dim=1, index=node_indices)
        mask_depot = (node_demands != 0).to(torch.float)

        depot_node_indice = depot_node_indice.gather(dim=1, index=node_indices)
        depot_node_indice = torch.cat((torch.zeros((depot_node_indice.size(0), 1), device=device, dtype=torch.long), depot_node_indice), dim=1)

        instances = (depot_coords, node_coords, node_demands, int(capacity))
        return instances, depot_node_indice, mask_depot, base_costs, valid_instance, bkp_data, rem_data

    else:
        bkp_data = None
        rem_data = (
            rem_xy_ls,
            rem_demand_ls,
            rem_indice_ls,
            rem_cost_ls,
        )
        return None, None, None, None, valid_instance, bkp_data, rem_data


def select_tail_data(bkp_data, new_data, bkp_cost, new_cost, device):
    assert bkp_cost.size() == new_cost.size()
    batch_size, _ = bkp_cost.size()
    beam_size = 1

    bkp_subind, bkp_cost_ls, bkp_ng_arr, bkp_xy, bkp_demand, bkp_indice, bkp_cap = bkp_data
    new_subind, new_cost_ls, new_ng_arr, new_xy, new_demand, new_indice, new_cap = new_data

    assert bkp_ng_arr.shape == new_ng_arr.shape == (batch_size, beam_size)
    assert bkp_subind.size(0) == bkp_cost_ls.size(0) == bkp_ng_arr.sum()
    assert new_subind.size(0) == new_cost_ls.size(0) == new_ng_arr.sum()

    bkp_graph_size, new_graph_size = bkp_xy.size(1), new_xy.size(1)
    bkp_subgraph_size, new_subgraph_size = bkp_subind.size(1), new_subind.size(1)
    assert bkp_xy.size(0) == new_xy.size(0) == batch_size
    assert bkp_demand.size() == bkp_indice.size() == (batch_size, bkp_graph_size)
    assert new_demand.size() == new_indice.size() == (batch_size, new_graph_size)

    graph_size = max(bkp_graph_size, new_graph_size)
    subgraph_size = max(bkp_subgraph_size, new_subgraph_size)

    if isinstance(bkp_cap, int):
        assert bkp_cap == new_cap
        capacity = bkp_cap
    else:
        raise NotImplementedError

    sel_subind, sel_cost_ls, sel_ng_arr, sel_xy, sel_demand, sel_indice = [], [], [], [], [], []
    bkp_cum_num = new_cum_num = 0
    subind_max_length = 0
    for i in range(batch_size):
        for j in range(beam_size):
            if bkp_cost[i, j] < new_cost[i, j]:
                sel_subind.append(bkp_subind[bkp_cum_num: bkp_cum_num + bkp_ng_arr[i, j]])
                sel_cost_ls.append(bkp_cost_ls[bkp_cum_num: bkp_cum_num + bkp_ng_arr[i, j]])
                sel_ng_arr.append(bkp_ng_arr[i, j])
                sel_xy.append(bkp_xy[i])
                sel_demand.append(bkp_demand[i])
                sel_indice.append(bkp_indice[i])
            else:
                sel_subind.append(new_subind[new_cum_num: new_cum_num + new_ng_arr[i, j]])
                sel_cost_ls.append(new_cost_ls[new_cum_num: new_cum_num + new_ng_arr[i, j]])
                sel_ng_arr.append(new_ng_arr[i, j])
                sel_xy.append(new_xy[i])
                sel_demand.append(new_demand[i])
                sel_indice.append(new_indice[i])
            subind_max_length = max(subind_max_length, sel_subind[-1].size(1))

            bkp_cum_num += bkp_ng_arr[i, j]
            new_cum_num += new_ng_arr[i, j]

    max_length = 0
    for i in range(batch_size):
        sel_sind = sel_subind[i]
        pad_num = subind_max_length - sel_sind.size(1)
        assert pad_num >= 0
        if pad_num > 0:
            tmp_sel_sind = []
            for j in range(len(sel_sind)):
                pad_idx = torch.where(sel_sind[j] == 0)[0][0]
                tmp_sel_sind.append(
                    torch.cat((sel_sind[j][:pad_idx], torch.zeros((pad_num,), device=device, dtype=torch.long), sel_sind[j][pad_idx:]), dim=0)
                )
            sel_subind[i] = torch.stack(tmp_sel_sind, dim=0)
        max_length = max(max_length, torch.unique(sel_subind[i].reshape((-1, ))).size(0))


    for i in range(batch_size):
        sel_sind = sel_subind[i]
        unq_sind, tmp_sind = torch.unique(sel_sind.reshape((-1,)), return_inverse=True)

        unq_sind = F.pad(unq_sind, (0, max_length - unq_sind.size(0)), mode='constant', value=0)
        sel_xy[i] = sel_xy[i].gather(index=unq_sind[:, None].repeat(1, 2), dim=0)
        sel_demand[i] = sel_demand[i].gather(index=unq_sind, dim=0)
        sel_indice[i] = sel_indice[i].gather(index=unq_sind, dim=0)
        sel_subind[i] = tmp_sind.reshape(sel_sind.size())

    sel_subind = torch.cat(sel_subind, dim=0)
    sel_ng_arr = np.array(sel_ng_arr).reshape((batch_size, 1))
    sel_cost_ls = torch.cat(sel_cost_ls, dim=0)
    sel_xy = torch.stack(sel_xy, dim=0)
    sel_demand = torch.stack(sel_demand, dim=0)
    sel_indice = torch.stack(sel_indice, dim=0)

    return sel_subind, sel_cost_ls, sel_ng_arr, sel_xy, sel_demand, sel_indice, capacity



def padding_for_taildata(selected_subinstances, selected_subdemands, selected_subindices, depot_xy, max_length):
    device = depot_xy.device
    batch_size, _ = depot_xy.size()
    assert batch_size == len(selected_subinstances) == len(selected_subdemands) == len(selected_subindices)

    padded_subinstances, padded_subdemands, padded_subindices, padded_ng_array = [], [], [], []
    for bc_idx in range(batch_size):
        sel_subinstances = selected_subinstances[bc_idx]
        sel_subdemands = selected_subdemands[bc_idx]
        sel_subindices = selected_subindices[bc_idx]

        n_subgraphs = len(sel_subinstances)
        assert n_subgraphs == len(sel_subdemands) == len(sel_subindices)
        depot = depot_xy[bc_idx:bc_idx+1]

        for sg_idx in range(n_subgraphs):
            pad_num = max_length - sel_subdemands[sg_idx].size(0)
            assert pad_num >= 0

            if pad_num > 0:
                pad_idx = torch.where(sel_subdemands[sg_idx] == 0)[0][0]
                assert sel_subindices[sg_idx][pad_idx] == 0
                assert (sel_subinstances[sg_idx][pad_idx] == depot[0]).all()

                padded_subinstances.append(
                    torch.cat((sel_subinstances[sg_idx][:pad_idx], depot.repeat(pad_num, 1), sel_subinstances[sg_idx][pad_idx:]), dim=0)
                )
                padded_subdemands.append(
                    torch.cat((sel_subdemands[sg_idx][:pad_idx], torch.zeros((pad_num, ), device=device, dtype=torch.int), sel_subdemands[sg_idx][pad_idx:]), dim=0)
                )
                padded_subindices.append(
                    torch.cat((sel_subindices[sg_idx][:pad_idx], torch.zeros((pad_num,), device=device, dtype=torch.long), sel_subindices[sg_idx][pad_idx:]), dim=0)
                )
            else:
                padded_subinstances.append(sel_subinstances[sg_idx])
                padded_subdemands.append(sel_subdemands[sg_idx])
                padded_subindices.append(sel_subindices[sg_idx])

        padded_ng_array.append(n_subgraphs)

    padded_subinstances = torch.stack(padded_subinstances, dim=0)
    padded_subdemands = torch.stack(padded_subdemands, dim=0)
    padded_subindices = torch.stack(padded_subindices, dim=0)
    padded_ng_array = np.array(padded_ng_array).reshape(batch_size, 1)

    return padded_subinstances, padded_subdemands, padded_subindices, padded_ng_array










def get_subinstances(depot_node_xy, depot_node_demand, selected_node_list, depot_idx_list, min_reviser_size=50, depot_node_indice=None):
    batch_size, problem_size, _ = depot_node_xy.size()
    _, beam_size, _ = selected_node_list.size()

    subgraphs = [
        [
            selected_node_list[i, j, depot_idx_list[i][j][k]: depot_idx_list[i][j][k + 1] + 1]
            for k in range(len(depot_idx_list[i][j]) - 1)
        ]
        for i in range(batch_size) for j in range(beam_size)
    ]
    n_graph_array = np.array([
        [len(depot_idx_list[i][j]) - 1 for j in range(beam_size)] for i in range(batch_size)
    ])

    max_subgraph_size = max([ max([seq[i].size(0) for i in range(len(seq))]) for seq in subgraphs])
    max_subgraph_size = max(max_subgraph_size, min_reviser_size)

    #n_depot = [int(max_subgraph_size - seq[i].size(0) + 2) for seq in subgraphs for i in range(len(seq))]

    padded_subgraphs = [
        torch.cat(
            [
                F.pad(seq[i], (0, max_subgraph_size - seq[i].size(0)), mode='constant', value=0)
                for i in range(len(seq))
            ],
            dim = 0
        )
        for seq in subgraphs
    ]

    subinstances = []
    subdemands = []
    subindices = []

    for i, seq in enumerate(padded_subgraphs):
        # coords: (problem, 2)
        # demand: (problem)
        # seq: (n, )
        coords = depot_node_xy[int(i // beam_size)]
        coords = coords.gather(dim=0, index=seq[:, None].expand(seq.size(0), coords.size(1)))
        coords = coords.reshape(-1, max_subgraph_size, 2)
        subinstances.append(coords)

        demands = depot_node_demand[int(i // beam_size)]
        demands = demands.gather(dim=0, index=seq)
        demands = demands.reshape(-1, max_subgraph_size)
        subdemands.append(demands)

        if depot_node_indice is not None:
            indices = depot_node_indice[int(i // beam_size)]
            indices = indices.gather(dim=0, index=seq)
            indices = indices.reshape(-1, max_subgraph_size)
            subindices.append(indices)
        else:
            indices = seq.reshape(-1, max_subgraph_size)
            subindices.append(indices)

    subinstances = torch.cat(subinstances, dim=0)
    subdemands = torch.cat(subdemands, dim=0)
    subindices = torch.cat(subindices, dim=0)
    return subinstances, subdemands, subindices, n_graph_array




@torch.no_grad()
def permute_partition(args, perm_models: List[Module], instances, demands, indices, n_graph_array, perm_model_type='am_reviser'):
    n_graphs, graph_size, _ = instances.size()
    device = args.perm_device

    instances = instances.to(device)
    demands = demands.to(device)
    indices = indices.to(device)

    if perm_model_type == 'am_reviser':
        insert_order = torch.arange(graph_size)
        solution = [random_insertion(inst, insert_order)[0] for inst in instances]
        solution = torch.tensor(np.array(solution).astype(np.int64), device=device).reshape(-1, graph_size)

        original_solution = torch.arange(graph_size, dtype=torch.long, device=device)
        original_cost = (instances - instances.roll(dims=1, shifts=1)).norm(p=2, dim=2).sum(1)

        updated_instances = instances.gather(dim=1, index=solution[:, :, None].expand_as(instances))
        updated_cost = (updated_instances - updated_instances.roll(dims=1, shifts=1)).norm(p=2, dim=2).sum(1)
        solution[updated_cost >= original_cost] = original_solution

        instances = instances.gather(dim=1, index=solution[:, :, None].expand_as(instances))
        demands = demands.gather(dim=1, index=solution)
        indices = indices.gather(dim=1, index=solution)

        instances, demands, indices, cost_list = reconnect(
            get_cost_func=get_cost_func,
            seed=instances,
            demands=demands,
            indices=indices,
            opts=args,
            revisers=perm_models,
        )
        assert cost_list.size(0) == instances.size(0) == n_graphs
        costs = restore_costs(cost_list, n_graph_array, args)

    elif perm_model_type == 'lkh3':
        cost_list, tours, duration = lkh_solve(args, instances.cpu().numpy().tolist())

        cost_list = torch.tensor(cost_list, device=device, dtype=torch.float)
        tours = torch.tensor(tours, device=device, dtype=torch.long)

        instances = instances.gather(dim=1, index=tours[:, :, None].expand_as(instances))
        demands = demands.gather(dim=1, index=tours)
        indices = indices.gather(dim=1, index=tours)
        costs = restore_costs(cost_list, n_graph_array, args)

    else:
        raise NotImplementedError

    return instances, demands, indices, cost_list, costs


def restore_costs(cost_list, n_graph_array, args):
    batch_size = n_graph_array.shape[0]
    beam_size = n_graph_array.shape[1]

    assert len(cost_list.size()) == 1
    assert cost_list.size(0) == n_graph_array.sum()

    cum_num = 0
    costs = []
    for i in range(batch_size):
        for j in range(beam_size):
            if args.min_max:
                costs.append(cost_list[cum_num: cum_num+n_graph_array[i, j]].max())
            else:
                costs.append(cost_list[cum_num: cum_num+n_graph_array[i, j]].sum())
            cum_num += n_graph_array[i, j]

    costs = torch.stack(costs).reshape(batch_size, beam_size)
    return costs


def reorder_instances_by_centroid(instances, n_graph_array, depot_coord):
    n_graphs, graph_size, _ = instances.size()
    batch_size, beam_size = n_graph_array.shape
    device = instances.device
    assert n_graph_array.sum() == n_graphs

    order = []
    cum_n_graphs = 0
    for i in range(batch_size):
        for j in range(beam_size):
            # (n_subinst, graph_size, 2)
            subinst = instances[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]]
            cum_n_graphs += n_graph_array[i, j]
            # (1, 1, 2)
            depot_xy = depot_coord[i][None, None, :]

            centroids = subinst.mean(dim=1, keepdim=True)
            idx = reorder_vector(centroids, depot_xy)
            order.append(idx)

    order = torch.cat(order, dim=0)
    return order




def reorder_instances_by_endpoint(instances, n_graph_array, depot_coord, direction):
    n_graphs, graph_size, _ = instances.size()
    batch_size, beam_size = n_graph_array.shape
    device = instances.device
    assert n_graph_array.sum() == n_graphs
    assert batch_size == depot_coord.size(0)

    fwd_order = []
    bwd_order = []
    cum_n_graphs = 0
    for i in range(batch_size):
        for j in range(beam_size):
            # (n_subinst, graph_size, 2)
            subinst = instances[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]]
            cum_n_graphs += n_graph_array[i, j]
            # (1, 1, 2)
            depot_xy = depot_coord[i][None, None, :]

            # (n_subinst, graph_size)
            is_other = ((subinst - depot_xy).norm(p=2, dim=2) > 1e-6)
            pos_diff = torch.logical_xor(is_other, is_other.roll(dims=1, shifts=1))
            neg_diff = torch.logical_xor(is_other, is_other.roll(dims=1, shifts=-1))

            left_bc, left_idx = torch.where(pos_diff * is_other)
            right_bc, right_idx = torch.where(neg_diff * is_other)

            # there may exist multiple circle in each subinstance.
            if left_idx.size(0) > n_graph_array[i, j]:
                is_dup = False
                dup_idx = []
                k = 0
                while True:
                    if k + 1 < left_bc.size(0):
                        if left_bc[k] == left_bc[k+1]:
                            dup_idx.append(k)
                            is_dup = True
                        else:
                            if is_dup:
                                dup_idx.append(k)
                                dup_idx = torch.tensor(dup_idx, dtype=torch.long, device=device)

                                bc_idx = torch.cat((left_bc[dup_idx], right_bc[dup_idx]), dim=0)
                                pt_idx = torch.cat((left_idx[dup_idx], right_idx[dup_idx]), dim=0)

                                min_idx, max_idx = outer_points(subinst[bc_idx, pt_idx], depot_xy[0])
                                left_idx[dup_idx[0]] = pt_idx[min_idx]
                                right_idx[dup_idx[0]] = pt_idx[max_idx]

                                left_bc = torch.cat((left_bc[:dup_idx[0]+1], left_bc[dup_idx[-1]+1:]), dim=0)
                                right_bc = torch.cat((right_bc[:dup_idx[0]+1], right_bc[dup_idx[-1]+1:]), dim=0)
                                left_idx = torch.cat((left_idx[:dup_idx[0]+1], left_idx[dup_idx[-1]+1:]), dim=0)
                                right_idx = torch.cat((right_idx[:dup_idx[0]+1], right_idx[dup_idx[-1]+1:]), dim=0)

                                k -= (dup_idx.size(0) - 1)

                                dup_idx = []
                                is_dup = False

                    else:
                        if is_dup:
                            dup_idx.append(k)
                            dup_idx = torch.tensor(dup_idx, dtype=torch.long, device=device)

                            bc_idx = torch.cat((left_bc[dup_idx], right_bc[dup_idx]), dim=0)
                            pt_idx = torch.cat((left_idx[dup_idx], right_idx[dup_idx]), dim=0)

                            min_idx, max_idx = outer_points(subinst[bc_idx, pt_idx], depot_xy[0])
                            left_idx[dup_idx[0]] = pt_idx[min_idx]
                            right_idx[dup_idx[0]] = pt_idx[max_idx]

                            left_bc = torch.cat((left_bc[:dup_idx[0] + 1], left_bc[dup_idx[-1] + 1:]), dim=0)
                            right_bc = torch.cat((right_bc[:dup_idx[0] + 1], right_bc[dup_idx[-1] + 1:]), dim=0)
                            left_idx = torch.cat((left_idx[:dup_idx[0] + 1], left_idx[dup_idx[-1] + 1:]), dim=0)
                            right_idx = torch.cat((right_idx[:dup_idx[0] + 1], right_idx[dup_idx[-1] + 1:]), dim=0)

                            k -= (dup_idx.size(0) - 1)

                            dup_idx = []
                            is_dup = False
                        break
                    k += 1
                assert (left_bc == right_bc).all() and (left_bc == torch.arange(n_graph_array[i, j], device=device)).all()

            assert left_idx.size(0) == right_idx.size(0) == n_graph_array[i, j]

            # (n_subinst, 2)
            idxes = torch.stack((left_idx, right_idx), dim=1)
            # (n_subinst, 2, 2)
            coords = torch.gather(subinst, dim=1, index=idxes[:, :, None].expand(n_graph_array[i, j], 2, subinst.size(-1)))
            # (n_subinst, )
            fwd_idx, bwd_idx = reorder_vector(coords, depot_xy)

            fwd_order.append(fwd_idx)
            bwd_order.append(bwd_idx)

    fwd_order = torch.cat(fwd_order, dim=0)
    bwd_order = torch.cat(bwd_order, dim=0)
    assert fwd_order.size(0) == bwd_order.size(0) == n_graphs

    if direction == 'forward':
        return fwd_order
    elif direction == 'backward':
        return bwd_order
    elif direction == 'bidirectional':
        return fwd_order, bwd_order




def reorder_vector(coords, origin):
    # coords: (batch, 1, 2) or (batch, 2, 2)
    # origin: (1, 1, 2)

    # (n_subinst, 1 or 2, 2)
    vector = coords - origin
    # (n_subinst, 1 or 2)
    angle = torch.arcsin(vector[:, :, 1] / vector.norm(p=2, dim=2))
    angle[(vector[:, :, 0] <= 0) * (vector[:, :, 1] > 0)] = math.pi - angle[
        (vector[:, :, 0] <= 0) * (vector[:, :, 1] > 0)]
    angle[(vector[:, :, 0] < 0) * (vector[:, :, 1] <= 0)] = math.pi - angle[
        (vector[:, :, 0] < 0) * (vector[:, :, 1] <= 0)]
    angle[(vector[:, :, 0] >= 0) * (vector[:, :, 1] < 0)] = 2 * math.pi + angle[
        (vector[:, :, 0] >= 0) * (vector[:, :, 1] < 0)]

    if angle.size(1) == 1:
        idx = torch.sort(angle[:, 0], descending=False)[1]
        return idx

    elif angle.size(1) == 2:
        angle_exchange = angle[:, 0] < angle[:, 1]
        angle[angle_exchange, 0], angle[angle_exchange, 1] = angle[angle_exchange, 1], angle[angle_exchange, 0]
        fwd_idx = torch.sort(angle[:, 0], descending=False)[1]
        bwd_idx = torch.sort(angle[:, 1], descending=True)[1]
        return fwd_idx, bwd_idx


def outer_points(points, origin):
    # points: (n_points, 2)
    # origin: (1, 2)
    n_points, _ = points.size()

    vector = points - origin
    angle = torch.arcsin(vector[:, 1] / vector.norm(p=2, dim=1))
    angle[(vector[:, 0] <= 0) * (vector[:, 1] > 0)] = math.pi - angle[(vector[:, 0] <= 0) * (vector[:, 1] > 0)]
    angle[(vector[:, 0] < 0) * (vector[:, 1] <= 0)] = math.pi - angle[(vector[:, 0] < 0) * (vector[:, 1] <= 0)]
    angle[(vector[:, 0] >= 0) * (vector[:, 1] < 0)] = 2 * math.pi + angle[(vector[:, 0] >= 0) * (vector[:, 1] < 0)]

    min_idx, max_idx = angle.min(dim=0)[1], angle.max(dim=0)[1]
    return min_idx, max_idx


def recover_solution(costs, subinstances, n_graph_array):
    batch_size, beam_size = n_graph_array.shape
    assert costs.shape == n_graph_array.shape
    assert beam_size == 1

    if isinstance(costs, torch.Tensor):
        costs = costs.cpu().numpy()
    if isinstance(subinstances, torch.Tensor):
        subinstances = subinstances.cpu().numpy()

    solutions = []
    cum_num = 0
    for i in range(batch_size):
        solutions.append((costs[i, 0], subinstances[cum_num: cum_num + n_graph_array[i, 0]]))
        cum_num +=  n_graph_array[i, 0]
    return solutions










