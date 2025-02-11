import copy
import sys

import torch
import math
import time

import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch import Tensor
import numpy as np

from typing import List, Any
from torch.nn import Module

from graph_partition.hlgp_sl.revision import reconnect
from insertion import random_insertion


@dataclass
class DecodingSubPb:
    """
    In decoding, we successively apply model on progressively smaller sub-problems.
    In each sub-problem, we keep track of the indices of each node in the original full-problem.
    """
    node_coords: Tensor
    demands: Tensor
    current_capacities: Tensor
    distance_matrices: Tensor
    original_idxs: Tensor
    mask_depot: Any



def get_random_instance(batch_size, problem_size, distribution, device):
    CAPACITIES = {
        20: 20,
        1000: 200,
        2000: 300,
        5000: 300,
        7000: 300,
    }
    depot_xy = torch.rand(size=(batch_size, 1, 2), device=device)
    node_xy = torch.rand(size=(batch_size, problem_size, 2), device=device)
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size), device=device)

    if problem_size in CAPACITIES:
        capacity = CAPACITIES[problem_size]
    else:
        capacity = math.ceil(30 + problem_size / 5) if problem_size >= 20 else 20
    #node_demand = node_demand / capacity
    return depot_xy, node_xy, node_demand, int(capacity)


def get_cos_sim_matrix(coords):
    dot_products = torch.matmul(coords, coords.transpose(1, 2))
    magnitude = coords.norm(p=2, dim=2, keepdim=True)
    magnitude_matrix = torch.matmul(magnitude, magnitude.transpose(1, 2))
    return dot_products / (magnitude_matrix + 1e-10)

def get_distance_matrix(coords):
    distance = (coords[:, :, None, :] - coords[:, None, :, :]).norm(p=2, dim=3)
    return distance


def get_adj_matrix(node_coords, cvrplib=False):
    shift_coords = node_coords - node_coords[:, 0:1, :]
    _x, _y = shift_coords[:, :, 0], shift_coords[:, :, 1]

    cos_sim_matrix = get_cos_sim_matrix(shift_coords)
    euc_dis_matrix = get_distance_matrix(shift_coords)

    if not cvrplib: return cos_sim_matrix, euc_dis_matrix

    cos_mat = (cos_sim_matrix + cos_sim_matrix.min()) / cos_sim_matrix.max()
    euc_mat = 1 - euc_dis_matrix
    return cos_mat + euc_mat, euc_dis_matrix




def eval_decoding(node_coords, node_demands, capacities, distance_matrix, part_model, beam_size, knns, mask_depot=None):
    if beam_size == 1:
        paths, via_depots, tour_lengths = greedy_decoding_loop(
            node_coords, node_demands, capacities, distance_matrix, part_model, knns, mask_depot
        )
    else:
        paths, via_depots, tour_lengths = beam_search_decoding_loop(
            node_coords, node_demands, capacities, distance_matrix, part_model, beam_size, knns, mask_depot
        )

    assert (paths.sum(dim=1) == paths.size(1) * (paths.size(1) - 1) / 2).all()
    return paths, via_depots, tour_lengths




def greedy_decoding_loop(node_coords, node_demands, capacities, distance_matrix, part_model, knns, mask_depot=None):
    batch_size, graph_size, _ = node_coords.size()
    device = node_coords.device

    original_idxs = torch.arange(graph_size, device=device)[None, :].repeat(batch_size, 1)
    initial_capacities = copy.deepcopy(capacities)
    sub_problem = DecodingSubPb(
        node_coords,
        node_demands,
        initial_capacities,
        distance_matrix,
        original_idxs,
        mask_depot,
    )

    paths = torch.zeros((batch_size, graph_size), dtype=torch.long, device=device)
    paths[:, -1] = graph_size - 1
    via_depots = torch.full((batch_size, graph_size), False, device=device, dtype=torch.bool)
    lengths = torch.zeros(batch_size, device=device)

    for step_idx in range(1, graph_size - 1):
        selected, via_depot, capacities, sub_problem = greedy_decoding_step(
            capacities, sub_problem, part_model, knns
        )

        paths[:, step_idx] = selected
        via_depots[:, step_idx] = via_depot

        lengths[~via_depot] += distance_matrix[~via_depot, paths[~via_depot, step_idx - 1], paths[~via_depot, step_idx]]
        lengths[via_depot] += distance_matrix[via_depot, paths[via_depot, step_idx-1], paths[via_depot, 0]] + \
                              distance_matrix[via_depot, paths[via_depot, step_idx], paths[via_depot, 0]]


    assert torch.count_nonzero(sub_problem.current_capacities < 0) == 0
    lengths += distance_matrix[torch.arange(batch_size), paths[:, -2], paths[:, -1]]

    return paths, via_depots, lengths




def greedy_decoding_step(capacities, sub_problem, part_model, knns):
    scores = prepare_input_and_forward_pass(capacities, sub_problem, part_model, knns)
    selected_nodes = torch.argmax(scores, dim=1, keepdim=True)
    idx_selected = torch.div(selected_nodes, 2, rounding_mode='trunc')
    via_depot = (selected_nodes % 2 == 1)
    idx_selected_original = torch.gather(sub_problem.original_idxs, dim=1, index=idx_selected)

    new_problem, via_depot = reformat_subproblem_for_next_step(
        capacities, sub_problem, idx_selected, via_depot, knns
    )
    return idx_selected_original.squeeze(1), via_depot.squeeze(1), capacities, new_problem



def beam_search_decoding_loop(node_coords, node_demands, capacities, distance_matrix, part_model, beam_size, knns, mask_depot=None):
    batch_size, graph_size, _ = node_coords.size()
    device = node_coords.device

    original_idxs = torch.arange(graph_size, device=device)[None, :].repeat(batch_size, 1)
    initial_capacities = copy.deepcopy(capacities)
    sub_problem = DecodingSubPb(
        node_coords,
        node_demands,
        initial_capacities,
        distance_matrix,
        original_idxs,
        mask_depot,
    )

    paths = torch.zeros((batch_size * beam_size, graph_size), dtype=torch.long, device=device)
    paths[:, -1] = graph_size - 1
    via_depots = torch.full((batch_size * beam_size, graph_size), False, device=device, dtype=torch.bool)
    lengths = torch.zeros(batch_size * beam_size, device=device)
    probabilities = torch.zeros((batch_size, 1), device=device)


    for step_idx in range(1, graph_size - 1):
        selected, via_depot, capacities, sub_problem, batch_in_prev_input, probabilities = \
            beam_search_decoding_step(capacities, sub_problem, part_model, probabilities, batch_size, beam_size, knns)

        paths = paths[batch_in_prev_input]
        via_depots = via_depots[batch_in_prev_input]
        distance_matrix = distance_matrix[batch_in_prev_input]
        lengths = lengths[batch_in_prev_input]

        paths[:, step_idx] = selected
        via_depots[:, step_idx] = via_depot

        lengths[~via_depot] += distance_matrix[~via_depot, paths[~via_depot, step_idx - 1], paths[~via_depot, step_idx]]
        lengths[via_depot] += distance_matrix[via_depot, paths[via_depot, step_idx - 1], paths[via_depot, 0]] + \
                              distance_matrix[via_depot, paths[via_depot, step_idx], paths[via_depot, 0]]

    lengths += distance_matrix[torch.arange(batch_size * beam_size), paths[:, -2], paths[:, -1]]

    lengths = lengths.reshape(batch_size, beam_size)
    paths = paths.reshape(batch_size, beam_size, graph_size)
    via_depots = via_depots.reshape(batch_size, beam_size, graph_size)

    min_lengths_idx = torch.argmin(lengths, dim=1)
    return paths[torch.arange(batch_size), min_lengths_idx], via_depots[torch.arange(batch_size), min_lengths_idx], \
           lengths[torch.arange(batch_size), min_lengths_idx]



def beam_search_decoding_step(capacities, sub_problem, part_model, probabilities, batch_size, beam_size, knns):
    graph_size = sub_problem.node_coords.shape[1]
    num_instances = sub_problem.node_coords.shape[0] // batch_size
    device = capacities.device

    scores = prepare_input_and_forward_pass(capacities, sub_problem, part_model, knns)
    candidates = torch.softmax(scores, dim=1)
    probabilities = (probabilities.repeat(1, 2 * graph_size) + candidates.log()).reshape(batch_size, -1)

    topk = min(beam_size, probabilities.shape[1] - 2)
    topk_values, topk_indexes = torch.topk(probabilities, topk, dim=1)

    batch_in_prev_input = (
            (num_instances * torch.arange(batch_size, device=device)).unsqueeze(dim=1) + \
            torch.div(topk_indexes, 2 * graph_size, rounding_mode='floor')
    ).flatten()
    topk_indexes = topk_indexes.flatten()
    topk_values = topk_values.flatten()

    sub_problem.node_coords = sub_problem.node_coords[batch_in_prev_input]
    sub_problem.original_idxs = sub_problem.original_idxs[batch_in_prev_input]
    sub_problem.demands = sub_problem.demands[batch_in_prev_input]
    sub_problem.current_capacities = sub_problem.current_capacities[batch_in_prev_input]
    sub_problem.distance_matrices = sub_problem.distance_matrices[batch_in_prev_input]
    capacities = capacities[batch_in_prev_input]

    if sub_problem.mask_depot != None:
        sub_problem.mask_depot = sub_problem.mask_depot[batch_in_prev_input]

    selected_nodes = torch.remainder(topk_indexes, 2 * graph_size).unsqueeze(dim=1)
    idx_selected = torch.div(selected_nodes, 2, rounding_mode='trunc')
    via_depot = (selected_nodes % 2 == 1)

    idx_selected_original = torch.gather(sub_problem.original_idxs, 1, idx_selected)
    new_subproblem, via_depot = reformat_subproblem_for_next_step(capacities, sub_problem, idx_selected, via_depot, knns)

    return idx_selected_original.squeeze(1), via_depot.squeeze(1), capacities, new_subproblem, batch_in_prev_input, topk_values.unsqueeze(1)







def prepare_input_and_forward_pass(capacities, sub_problem, net, knns):
    # find K nearest neighbors of the current node
    bs, num_nodes, node_dim = sub_problem.node_coords.shape
    if 0 < knns < num_nodes:
        bs, num_nodes, node_dim = sub_problem.node_coords.shape
        device = sub_problem.node_coords.device

        # (batch, knns)
        knn_indices = torch.topk(sub_problem.distance_matrices[:, :-1, 0], k=knns - 1, dim=-1, largest=False).indices
        knn_indices = torch.cat([knn_indices, torch.full([bs, 1], num_nodes - 1, device=device)], dim=-1)

        # (batch, knns, 2)
        knn_coords = torch.gather(sub_problem.node_coords, 1, knn_indices.unsqueeze(dim=-1).repeat(1, 1, node_dim))
        # (batch, knns)
        knn_demands = torch.gather(sub_problem.demands, 1, knn_indices)
        # (batch, )
        current_capacities = sub_problem.current_capacities[:, -1]

        mask_depot = None
        if sub_problem.mask_depot != None:
            assert sub_problem.mask_depot.size() == sub_problem.demands.size()
            mask_depot = sub_problem.mask_depot.gather(dim=1, index=knn_indices)

        inputs = torch.cat([
            knn_coords,
            (knn_demands / capacities).unsqueeze(-1),
            (current_capacities.unsqueeze(-1) / capacities).repeat(1, knns).unsqueeze(-1)
        ], dim=-1)

        knn_scores = net(inputs, demands=knn_demands, remaining_capacities=current_capacities, mask_depot=mask_depot)  # (b, seq)

        # create result tensor for scores with all -inf elements
        scores = torch.full(
            (sub_problem.node_coords.shape[0], 2 * sub_problem.node_coords.shape[1]), -np.inf, device=device
        )
        double_knn_indices = torch.zeros(
            [knn_indices.shape[0], 2 * knn_indices.shape[1]], device=device, dtype=torch.int64
        )
        double_knn_indices[:, 0::2] = 2 * knn_indices
        double_knn_indices[:, 1::2] = 2 * knn_indices + 1

        # and put computed scores for KNNs
        scores = torch.scatter(scores, 1, double_knn_indices, knn_scores)

    else:
        current_capacities = sub_problem.current_capacities[:, -1]
        inputs = torch.cat([
            sub_problem.node_coords,
            (sub_problem.demands / capacities).unsqueeze(-1),
            (current_capacities.unsqueeze(-1) / capacities).repeat(1, num_nodes).unsqueeze(-1)
        ], dim=-1)
        scores = net(inputs, demands=sub_problem.demands, remaining_capacities=current_capacities, mask_depot=sub_problem.mask_depot)

    return scores



def reformat_subproblem_for_next_step(capacities, sub_problem, idx_selected, via_depot, knns):
    # Example: current_subproblem: [a b c d e] => (model selects d) => next_subproblem: [d b c e]
    bs, subpb_size, _ = sub_problem.node_coords.shape
    device = sub_problem.node_coords.device

    # (bs, subp_size)
    is_selected = torch.arange(subpb_size, device=device).unsqueeze(dim=0).repeat(bs, 1) == idx_selected.repeat(1, subpb_size)

    # next begin node = just-selected node
    # (bs, 1, 2) or (bs, 1)
    next_begin_node_coord = sub_problem.node_coords[is_selected].unsqueeze(dim=1)
    next_begin_demand = sub_problem.demands[is_selected].unsqueeze(dim=1)
    next_begin_original_idx = sub_problem.original_idxs[is_selected].unsqueeze(dim=1)

    # remaining nodes = the rest, minus current first node
    # (bs, subp_size - 2, 2) or (bs, subp_size - 2)
    next_remaining_node_coords = sub_problem.node_coords[~is_selected].reshape((bs, -1, 2))[:, 1:, :]
    next_remaining_demands = sub_problem.demands[~is_selected].reshape((bs, -1))[:, 1:]
    next_remaining_original_idxs = sub_problem.original_idxs[~is_selected].reshape((bs, -1))[:, 1:]

    # concatenate
    # (bs, subp_size - 1, 2) or (bs, subp_size - 1)
    next_node_coords = torch.cat([next_begin_node_coord, next_remaining_node_coords], dim=1)
    next_demands = torch.cat([next_begin_demand, next_remaining_demands], dim=1)
    next_original_idxs = torch.cat([next_begin_original_idx, next_remaining_original_idxs], dim=1)

    # update current capacities
    # (bs, 1)
    current_capacities = sub_problem.current_capacities[:, -1].unsqueeze(dim=1) - next_begin_demand
    # recompute capacities
    current_capacities[via_depot.bool()] = capacities[via_depot.bool()] - next_begin_demand[via_depot.bool()]

    if torch.count_nonzero(current_capacities < 0) > 0:
        print("stp")

    next_mask_depot = None
    if sub_problem.mask_depot != None:
        next_begin_mask_depot = sub_problem.mask_depot[is_selected].unsqueeze(dim=1)
        next_remaining_mask_depot = sub_problem.mask_depot[~is_selected].reshape((bs, -1))[:, 1:]
        next_mask_depot = torch.cat([next_begin_mask_depot, next_remaining_mask_depot], dim=1)


    next_current_capacities = torch.cat([sub_problem.current_capacities, current_capacities], dim=-1)
    if knns != -1:
        num_nodes = sub_problem.distance_matrices.shape[1]

        """
        # select row (=column) of adj matrix for just-selected node
        # (bs, subp_size)
        next_row_column = sub_problem.distance_matrices[is_selected]
        # remove distance to the selected node (=0)
        # (bs, subp_size - 2)
        next_row_column = next_row_column[~is_selected].reshape((bs, -1))[:, 1:]

        # remove rows and columns of selected nodes
        # (bs, subp_size - 2, subp_size)
        next_adj_matrices = sub_problem.distance_matrices[~is_selected].reshape(bs, -1, num_nodes)[:, 1:, :]
        # (bs, subp_size - 2, subp_size - 2)
        next_adj_matrices = next_adj_matrices.transpose(1, 2)[~is_selected].reshape(bs, num_nodes-1, -1)[:, 1:, :]

        # add new row on the top and remove second (must be done like this, because on dimenstons of the matrix)
        # (bs, subp_size - 1, subp_size - 2)
        next_adj_matrices = torch.cat([next_row_column.unsqueeze(dim=1), next_adj_matrices], dim=1)

        # and add it to the beginning-
        # (bs, subp_size - 1)
        next_row_column = torch.cat([torch.zeros(idx_selected.shape, device=device), next_row_column], dim=1)
        # (bs, subp_size - 1, subp_size - 1)
        next_adj_matrices = torch.cat([next_row_column.unsqueeze(dim=2), next_adj_matrices], dim=2)
        """
        next_dis_matrices = reformat_subproblem_matrices(bs, is_selected, idx_selected, sub_problem.distance_matrices)
    else:
        next_dis_matrices = sub_problem.distance_matrices

    new_subproblem = DecodingSubPb(
        next_node_coords, next_demands, next_current_capacities,
        next_dis_matrices, next_original_idxs, next_mask_depot,
    )

    return new_subproblem, via_depot



def reformat_subproblem_matrices(bs, is_selected, idx_selected, subp_matrices):
    num_nodes = subp_matrices.shape[1]
    device = is_selected.device
    #num_nodes = sub_problem.distance_matrices.shape[1]

    # select row (=column) of adj matrix for just-selected node
    # (bs, subp_size)
    next_row_column = subp_matrices[is_selected]
    #next_row_column = sub_problem.distance_matrices[is_selected]
    # remove distance to the selected node (=0)
    # (bs, subp_size - 2)
    next_row_column = next_row_column[~is_selected].reshape((bs, -1))[:, 1:]

    # remove rows and columns of selected nodes
    # (bs, subp_size - 2, subp_size)
    next_matrices = subp_matrices[~is_selected].reshape(bs, -1, num_nodes)[:, 1:, :]
    #next_adj_matrices = sub_problem.distance_matrices[~is_selected].reshape(bs, -1, num_nodes)[:, 1:, :]
    # (bs, subp_size - 2, subp_size - 2)
    next_matrices = next_matrices.transpose(1, 2)[~is_selected].reshape(bs, num_nodes - 1, -1)[:, 1:, :]
    #next_adj_matrices = next_adj_matrices.transpose(1, 2)[~is_selected].reshape(bs, num_nodes - 1, -1)[:, 1:, :]

    # add new row on the top and remove second (must be done like this, because on dimenstons of the matrix)
    # (bs, subp_size - 1, subp_size - 2)
    next_matrices = torch.cat([next_row_column.unsqueeze(dim=1), next_matrices], dim=1)
    #next_adj_matrices = torch.cat([next_row_column.unsqueeze(dim=1), next_adj_matrices], dim=1)

    # and add it to the beginning-
    # (bs, subp_size - 1)
    next_row_column = torch.cat([torch.zeros(idx_selected.shape, device=device), next_row_column], dim=1)

    # (bs, subp_size - 1, subp_size - 1)
    next_matrices = torch.cat([next_row_column.unsqueeze(dim=2), next_matrices], dim=2)
    #next_adj_matrices = torch.cat([next_row_column.unsqueeze(dim=2), next_adj_matrices], dim=2)
    return next_matrices


def recover_solution_from_subindice(subindices, n_graph_array):
    n_subgraphs, subgraph_size = subindices.size()
    batch_size, beam_size = n_graph_array.shape
    assert beam_size == 1
    assert n_graph_array.sum() == n_subgraphs

    cum_num = 0
    solutions = []
    for i in range(batch_size):
        for j in range(beam_size):
            subind = subindices[cum_num: cum_num + n_graph_array[i, j]]

            rotated_subind = []
            for k in range(subind.size(0)):
                is_other = (subind[k] != 0)
                is_border = torch.logical_xor(is_other, is_other.roll(dims=0, shifts=1))
                depot_right = torch.where(is_other * is_border)[0]

                ind = subind[k].roll(dims=0, shifts=int(1 - depot_right[0].item()))
                if len(torch.where(ind == 0)[0]) >= 3:
                    ind = ind[0:torch.where(ind == 0)[0][2]]
                assert ind.sum() == subind[k].sum()
                assert ind[0] == ind[-1] == 0
                assert ind[1] != 0 and ind[-2] != 0
                rotated_subind.append(ind[1:-1])

            solutions.append(rotated_subind)
            cum_num += n_graph_array[i, j]

    return solutions


def check_subindices(subindices):
    n_subgraphs, subgraph_size = subindices.size()
    invalid = False
    for sg_idx in range(n_subgraphs):
        subind = subindices[sg_idx]

        is_other = (subind != 0)
        is_border = torch.logical_xor(is_other, is_other.roll(dims=0, shifts=1))
        depot_right = torch.where(is_other * is_border)[0]

        ind = subind.roll(dims=0, shifts=int(1 - depot_right[0].item()))
        if len(torch.where(ind == 0)[0]) >= 3:
            ind = ind[0:torch.where(ind == 0)[0][2]]
        try:
            assert ind.sum() == subind.sum()
            assert ind[0] == ind[-1] == 0
            assert torch.where(ind[1:-1] == 0)[0].size(0) == 0
        except:
            invalid = True
            print(sg_idx)
    return invalid




def prepare_training_instances(train_instances, solutions, reorder_mode=None):
    depot_coords, node_coords, node_demands, capacity = train_instances
    batch_size, problem_size, _ = node_coords.size()
    device = node_coords.device
    assert len(solutions) == batch_size

    depot_demands = torch.zeros((batch_size, 1), dtype=torch.int, device=device)
    node_demands = torch.cat((depot_demands, node_demands, depot_demands), dim=1)
    node_coords = torch.cat((depot_coords, node_coords, depot_coords), dim=1)

    coordinates, demands, remaining_capacities, via_depots = [], [], [], []
    for i in range(batch_size):
        if reorder_mode == 'sub_solution':
            coords, dems, rem_caps, via_depot = \
                reorder_by_subsolution(node_coords[i], node_demands[i], capacity, solutions[i])
        elif reorder_mode == 'full_solution':
            coords, dems, rem_caps, via_depot = \
                reorder_by_fullsolution(node_coords[i], node_demands[i], capacity, solutions[i][1])

        coordinates.append(coords)
        demands.append(dems)
        remaining_capacities.append(rem_caps)
        via_depots.append(via_depot)

    coordinates = torch.stack(coordinates)
    demands = torch.stack(demands)
    remaining_capacities = torch.stack(remaining_capacities)
    via_depots = torch.stack(via_depots)
    capacities = torch.full(size=(batch_size, ), dtype=torch.float, device=device, fill_value=capacity)

    return coordinates, demands, capacities, remaining_capacities, via_depots




def reorder_by_subsolution(coords, demands, capacity, solution):
    capacities = torch.stack([capacity - demands.gather(dim=0, index=tour).sum() for tour in solution])
    tour_idxs = torch.argsort(capacities)

    reformated_solution, remaining_capacities = [], []
    for t_idx in tour_idxs:
        reformated_solution.append(solution[t_idx])
        remaining_capacities.append(capacity - demands.gather(dim=0, index=solution[t_idx]).cumsum(dim=0))

    reformated_solution = torch.cat(reformated_solution, dim=0)
    remaining_capacities = torch.cat(remaining_capacities, dim=0)

    reformated_solution = F.pad(reformated_solution, pad=[1, 1], mode='constant', value=0)
    remaining_capacities = F.pad(remaining_capacities, pad=[1, 1], mode='constant', value=capacity)

    via_depot = torch.zeros_like(reformated_solution)
    depot_idxs = torch.where(remaining_capacities[1:] > remaining_capacities[:-1])[0] + 1
    via_depot[depot_idxs] = 1

    coords = torch.gather(coords, dim=0, index=reformated_solution[:, None].repeat(1, 2))
    demands = torch.gather(demands, dim=0, index=reformated_solution)

    return coords, demands, remaining_capacities, via_depot


def reorder_by_fullsolution(coords, demands, capacity, solution):
    solution = [0] + solution + [0]
    solution = torch.tensor(solution, dtype=torch.long, device=coords.device)
    depot_idxs = torch.where(solution == 0)[0].cpu().numpy().tolist()
    solution_list = []
    for i in range(len(depot_idxs) - 1):
        solution_list.append(solution[depot_idxs[i]+1:depot_idxs[i+1]])

    coords, dems, rem_caps, via_depot = reorder_by_subsolution(coords, demands, capacity, solution_list)
    return coords, dems, rem_caps, via_depot




def prepare_batch_data(data, sample=True):
    ks = '_s' if sample else ''
    node_coords = data[f"node_coords{ks}"]
    demands = data[f"node_demands{ks}"]
    capacities = data[f"capacities{ks}"]
    remaining_capacities = data[f"remaining_capacities{ks}"]
    via_depots = data[f"via_depots{ks}"]
    distance_matrices = data[f"dist_matrices{ks}"]

    return node_coords, demands, capacities, remaining_capacities, via_depots, distance_matrices




def get_cost_func(sol_coord, tour):
    return (sol_coord[:, 1:] - sol_coord[:, :-1]).norm(p=2, dim=2).sum(1)



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
        costs = restore_costs(cost_list, n_graph_array)

    elif perm_model_type == 'lkh3':
        cost_list, tours, duration = lkh_solve(args, instances.cpu().numpy().tolist())

        cost_list = torch.tensor(cost_list, device=device, dtype=torch.float)
        tours = torch.tensor(tours, device=device, dtype=torch.long)

        instances = instances.gather(dim=1, index=tours[:, :, None].expand_as(instances))
        demands = demands.gather(dim=1, index=tours)
        indices = indices.gather(dim=1, index=tours)
        costs = restore_costs(cost_list, n_graph_array)

    else:
        raise NotImplementedError

    return instances, demands, indices, cost_list, costs


def restore_costs(cost_list, n_graph_array):
    batch_size = n_graph_array.shape[0]
    beam_size = n_graph_array.shape[1]

    assert len(cost_list.size()) == 1
    assert cost_list.size(0) == n_graph_array.sum()

    cum_num = 0
    costs = []
    for i in range(batch_size):
        for j in range(beam_size):
            costs.append(cost_list[cum_num: cum_num+n_graph_array[i, j]].sum())
            cum_num += n_graph_array[i, j]

    costs = torch.stack(costs).reshape(batch_size, beam_size)
    return costs



def revise_partition(instances, demands, indices, cost_list, n_graph_array, depot_coord, capacity, part_model, perm_model, args, beam_size, n_neigh=2):
    part_model.eval()
    device = args.loc_part_device
    revision_part_iters = args.eval_revision_iters

    instances = instances.to(device)
    demands = demands.to(device)
    indices = indices.to(device)
    cost_list = cost_list.to(device)
    depot_coord = depot_coord.to(device)

    for i in range(revision_part_iters):
        shifts = 0 if i == 0 else 1

        instances, demands, indices, cost_list, n_graph_array = local_partition(
            instances, demands, indices, cost_list, n_graph_array, depot_coord, capacity,
            part_model, perm_model, args, direction=args.revise_direction, shifts=shifts,
            beam_size=beam_size, use_mode='validation', n_neigh=n_neigh,
        )

    costs = restore_costs(cost_list, n_graph_array)
    return instances, demands, indices, cost_list, costs, n_graph_array


def prepare_for_bq_input(instances):
    batch_size, problem_size, _ = instances[1].size()
    device = instances[1].device

    depot_coords, node_coords, node_demands, capacities = instances
    depot_demands = torch.zeros((batch_size, 1), dtype=torch.int, device=device)

    node_coords = torch.cat((depot_coords, node_coords, depot_coords), dim=1)
    node_demands = torch.cat((depot_demands, node_demands, depot_demands), dim=1)
    capacities = torch.full(size=(batch_size, 1), fill_value=capacities, dtype=torch.int, device=device)
    distance_matrix = get_distance_matrix(node_coords)
    return node_coords, node_demands, capacities, distance_matrix



def local_partition(instances, demands, indices, cost_list, n_graph_array, depot_coord, capacity,
                    part_model, perm_model, args, direction='forward', shifts=0, beam_size=None,
                    use_mode='validation', n_neigh=2, perm_model_type='am_reviser'):

    perm_model = [perm_model] if not isinstance(perm_model, List) else perm_model
    assert n_graph_array.shape[1] == 1

    device = args.loc_part_device
    problem_size = int(n_neigh * instances.size(1))

    order = reorder_instances_by_endpoint(instances, n_graph_array, depot_coord, direction=direction)
    neigh_instances, neigh_cost, neigh_indices, neigh_max_vehicle, neigh_mask_depot = construct_local_dataset(
        instances, demands, indices, cost_list, n_graph_array, depot_coord, order=order, shifts=shifts, replace=True, n_neigh=n_neigh
    )
    neigh_graph_array = np.ceil(n_graph_array / n_neigh).astype(int)
    neigh_indices = torch.cat((torch.zeros(neigh_indices.size(0), 1, device=device, dtype=torch.long), neigh_indices), dim=1)
    neigh_mask_depot = torch.cat((torch.ones(neigh_mask_depot.size(0), 1, device=device, dtype=torch.int), neigh_mask_depot,
                                  torch.ones(neigh_mask_depot.size(0), 1, device=device, dtype=torch.int)), dim=1)

    node_coords, node_demands, capacities, distance_matrix = prepare_for_bq_input(neigh_instances + (capacity, ))
    paths, via_depots, tour_lengths = eval_decoding(
        node_coords, node_demands, capacities, distance_matrix, part_model, beam_size, args.knns, mask_depot=neigh_mask_depot
    )
    selected_node_list, _ = reconstruct_tours(paths, via_depots)
    selected_node_list, depot_idx_list = remove_invalid_nodes(selected_node_list, neigh_indices)

    new_instances, new_demands, new_indices, new_ng_array = get_subinstances(
        node_coords[:, :-1], node_demands[:, :-1], selected_node_list, depot_idx_list, depot_node_indice=neigh_indices
    )
    new_instances, new_demands, new_indices, new_cost_list, new_cost = permute_partition(
        args, perm_model, new_instances, new_demands, new_indices, new_ng_array, perm_model_type=perm_model_type
    )

    ori_ng_array = neigh_max_vehicle.cpu().numpy()[:, None]
    assert neigh_graph_array.sum() == ori_ng_array.shape[0] == new_ng_array.shape[0]


    selected_instances, selected_demands, selected_indices, selected_cost, n_graph_array = select_partition_from_baseline(
        new_instances, new_demands, new_indices, new_cost, new_ng_array,
        instances, demands, indices, neigh_cost, ori_ng_array, neigh_instances, neigh_graph_array
    )

    return selected_instances, selected_demands, selected_indices, selected_cost, n_graph_array





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





def construct_local_dataset(instances, demands, indices, cost_list, n_graph_array, depot_coord, order=None, shifts=0, replace=False, n_neigh=2):


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
            mask_depot = ((neigh_coord[-1] - depot_coord[i:i+1]).norm(p=2, dim=2) <= 1e-6).to(torch.int)

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

    return (neigh_depot, neigh_coord, neigh_demand), neigh_cost.sum(dim=-1, keepdim=True), neigh_indice, max_vehicle_list, mask_depot_list






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
            subinst = instances[cum_n_graphs: cum_n_graphs + n_graph_array[i, j]]
            cum_n_graphs += n_graph_array[i, j]
            depot_xy = depot_coord[i][None, None, :]

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

            idxes = torch.stack((left_idx, right_idx), dim=1)
            coords = torch.gather(subinst, dim=1, index=idxes[:, :, None].expand(n_graph_array[i, j], 2, subinst.size(-1)))
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
    vector = coords - origin
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
    n_points, _ = points.size()

    vector = points - origin
    angle = torch.arcsin(vector[:, 1] / vector.norm(p=2, dim=1))
    angle[(vector[:, 0] <= 0) * (vector[:, 1] > 0)] = math.pi - angle[(vector[:, 0] <= 0) * (vector[:, 1] > 0)]
    angle[(vector[:, 0] < 0) * (vector[:, 1] <= 0)] = math.pi - angle[(vector[:, 0] < 0) * (vector[:, 1] <= 0)]
    angle[(vector[:, 0] >= 0) * (vector[:, 1] < 0)] = 2 * math.pi + angle[(vector[:, 0] >= 0) * (vector[:, 1] < 0)]

    min_idx, max_idx = angle.min(dim=0)[1], angle.max(dim=0)[1]
    return min_idx, max_idx


def reconstruct_tours(paths, via_depots):
    assert len(paths.shape) == len(via_depots.shape)
    beam_size = 1
    device = via_depots.device

    if len(paths.shape) == 3:
        beam_size = paths.shape[1]
        paths = paths.reshape(-1, paths.shape[-1])
        via_depots = via_depots.reshape(-1, via_depots[-1])

    batch_size, graph_size = paths.shape
    complete_paths = [[0] for _ in range(batch_size)]
    depot_idx_list = [[[0] for _ in range(beam_size)] for _ in range(int(batch_size // beam_size))]

    for step_idx in range(1, paths.shape[1] - 1):
        nodes_to_add = paths[:, step_idx].tolist()
        for bc_idx in (via_depots[:, step_idx] == True).nonzero().squeeze(-1).cpu().numpy():
            complete_paths[bc_idx].append(0)
            depot_idx_list[int(bc_idx // beam_size)][int(bc_idx % beam_size)].append(len(complete_paths[bc_idx]) - 1)

        for bc_idx in range(batch_size):
            complete_paths[bc_idx].append(nodes_to_add[bc_idx])

    for bc_idx in range(batch_size):
        if complete_paths[bc_idx][-1] != 0:
            complete_paths[bc_idx].append(0)
            depot_idx_list[int(bc_idx // beam_size)][int(bc_idx % beam_size)].append(len(complete_paths[bc_idx]) - 1)

    max_length = max([len(path) for path in complete_paths])
    complete_paths = [
        F.pad(torch.tensor(path, device=device), pad=[0, max_length - len(path)], mode='constant', value=0.)
        for path in complete_paths
    ]
    complete_paths = torch.stack(complete_paths, dim=0).to(torch.long).reshape(int(batch_size // beam_size), beam_size, -1)

    return complete_paths, depot_idx_list



def remove_invalid_nodes(selected_node_list, neigh_indices):
    batch_size, beam_size, graph_size = selected_node_list.size()
    assert neigh_indices.size(0) == batch_size

    neigh_indices = neigh_indices[:, None, :].repeat(1, beam_size, 1)
    original_idxs = torch.gather(neigh_indices, dim=2, index=selected_node_list)
    selected_node_list[original_idxs == 0] = 0

    depot_idx_list = [[[0] for _ in range(beam_size)] for _ in range(int(batch_size // beam_size))]
    for i in range(batch_size):
        for j in range(beam_size):
            selected_node = selected_node_list[i, j]
            for k in range(1, len(selected_node)):
                if selected_node[k] == 0 and selected_node[k-1] != 0:
                    depot_idx_list[i][j].append(k)
    return selected_node_list, depot_idx_list



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








