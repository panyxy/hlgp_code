import math

import numpy as np
import torch
from dataclasses import dataclass
import os, sys
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)


@dataclass
class Reset_State:
    depot_node_xy_demand: torch.Tensor = None
    depot_node_polar_demand: torch.Tensor = None
    edge_attr: torch.Tensor = None
    edge_index: torch.Tensor = None



def get_random_instance(batch_size, problem_size, distribution, device):
    CAPACITIES = {
        20: 20,
        1000: 200,
        2000: 300,
        5000: 300,
        7000: 300,
        10000: 300,
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



def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems


class OneShotCVRPEnv:
    def __init__(self,
                 args,
                 batch_size,
                 problem_size,
                 beam_size,
                 device,
                 ):
        self.args = args
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.problem_size = problem_size
        self.device = device

        self.depot_node_xy = None
        self.depot_node_demand = None
        self.capacity = None
        self.total_demand = None
        self.max_vehicle = None

        self.selected_count = None
        self.cur_node = None
        self.selected_node_list = None
        self.depot_idx_list = None

        self.vehicle_count = None
        self.cur_capacity = None
        self.demand_count = None

        self.visited_mask = None
        self.capacity_mask = None
        self.depot_mask = None

        self.cos_sim_matrix = None
        self.euc_dis_matrix = None
        self.topK = min(problem_size + 1, args.gnn_topK)

        self.reset_state = Reset_State()

    def load_problems(self, problems, aug_factor=1):
        if problems is not None:
            depot_xy, node_xy, node_demand, capacity = problems
        else:
            depot_xy, node_xy, node_demand, capacity = get_random_instance(
                self.batch_size, self.problem_size, distribution='uniform', device=self.device
            )

        if aug_factor == 8:
            self.batch_size = self.batch_size * 8
            depot_xy = augment_xy_data_by_8_fold(depot_xy)
            node_xy = augment_xy_data_by_8_fold(node_xy)
            node_demand = node_demand.repeat(8, 1)

        depot_demand = torch.zeros(size=(self.batch_size, 1), device=self.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1).to(torch.int)
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        self.capacity = capacity

        self.reset_state.depot_node_xy_demand = torch.cat(
            (self.depot_node_xy, (self.depot_node_demand / capacity)[:, :, None]), dim=2
        )

    def reset(self, beam_size=None, max_vehicle=None, mask_depot=None):
        self.beam_size = beam_size if beam_size is not None else self.beam_size

        batch_size = self.batch_size
        beam_size = self.beam_size
        problem_size = self.problem_size
        device = self.device

        self.total_demand = self.depot_node_demand.sum(dim=1, keepdim=True).repeat(1, self.beam_size)

        if max_vehicle is None:
            self.max_vehicle = (torch.ceil(self.total_demand / self.capacity) + 1.0).to(torch.int)
        else:
            if isinstance(max_vehicle, int):
                self.max_vehicle = torch.full(
                    size=(self.batch_size, self.beam_size), dtype=torch.int, device=self.device, fill_value=max_vehicle
                )
            elif isinstance(max_vehicle, torch.Tensor):
                self.max_vehicle = max_vehicle[:, None].expand(self.batch_size, self.beam_size)



        self.selected_count = 0
        self.cur_node = torch.zeros(size=(batch_size, beam_size, 1), dtype=torch.long, device=device)
        self.selected_node_list = torch.zeros(size=(batch_size, beam_size, 0), dtype=torch.long, device=device)
        self.depot_idx_list = [ [ [0] for j in range(beam_size) ] for i in range(batch_size) ]

        self.vehicle_count = torch.zeros(size=(batch_size, beam_size), dtype=torch.int, device=device)
        self.demand_count = torch.zeros(size=(batch_size, beam_size), dtype=torch.int, device=device)
        self.cur_capacity = torch.ones(size=(batch_size, beam_size), dtype=torch.int, device=device) * self.capacity

        self.visited_mask = torch.ones(size=(batch_size, beam_size, 1+problem_size), device=device)
        self.capacity_mask = torch.ones(size=(batch_size, beam_size, 1+problem_size), device=device)
        self.depot_mask = torch.ones(size=(batch_size, beam_size, 1+problem_size), device=device)
        self.segment_mask = torch.ones(size=(batch_size, beam_size, 1 + problem_size), device=device)


        if mask_depot != None:
            self.visited_mask[:, :, 1:] = self.visited_mask[:, :, 1:] * mask_depot[:, None, :]

        self.cos_sim_matrix = self.get_cos_sim_matrix(self.depot_node_xy)
        self.euc_dis_matrix = self.get_distance_matrix(self.depot_node_xy)

        depot_node_polar_demand, edge_attr, edge_index = self.gen_pyg_data(self.topK)
        self.reset_state.depot_node_polar_demand = depot_node_polar_demand
        self.reset_state.edge_attr = edge_attr
        self.reset_state.edge_index = edge_index

        return self.reset_state



    def step(self, heatmap, sampling_type, picked_nodes=None):
        # heatmap: (batch, graph, graph)
        batch_size = self.batch_size
        beam_size = self.beam_size
        problem_size = self.problem_size
        device = self.device

        log_prob_list = torch.zeros(size=(batch_size, beam_size, 1), device=device)

        self.selected_count += 1
        self.cur_node = torch.zeros(size=(batch_size, beam_size, 1), dtype=torch.long, device=device)
        self.selected_node_list = torch.cat((self.selected_node_list, self.cur_node), dim=2)

        global_dones = self.global_dones()
        while not global_dones.all():

            self.update_demand_state(self.cur_node)
            self.update_visited_mask(self.cur_node)
            self.update_capacity_mask()
            self.update_depot_mask()

            selection = None
            if picked_nodes is not None:
                selection = picked_nodes[:, :, self.selected_count:self.selected_count + 1]

            self.cur_node, log_prob = self.pick_node(
                heatmap, self.cur_node, sampling_type, selection=selection
            )
            self.selected_count += 1
            self.selected_node_list = torch.cat((self.selected_node_list, self.cur_node), dim=2)
            log_prob_list = torch.cat((log_prob_list, log_prob), dim=2)

            self.update_depot_idx(global_dones, self.global_dones(), self.cur_node, self.selected_count)
            global_dones = self.global_dones()

        return log_prob_list


    def pick_node(self, heatmap, cur_node, sampling_type, selection=None,):
        # heatmap: (batch, graph, graph)
        # cur_node: (batch, beam, 1)
        batch_size = self.batch_size
        beam_size = self.beam_size
        problem_size = self.problem_size
        device = self.device

        next_node_probs = heatmap.gather(dim=1, index=cur_node.expand(batch_size, beam_size, 1 + problem_size))
        step_mask = (self.visited_mask * self.capacity_mask * self.depot_mask * self.segment_mask).detach()
        assert (step_mask.sum(dim=2) >= 1).all()

        probs = (next_node_probs * step_mask).reshape(-1, 1 + problem_size)
        dist = torch.distributions.Categorical(probs)

        if sampling_type == 'sampling':
            next_node = dist.sample()
            log_prob = dist.log_prob(next_node)
        elif sampling_type == 'greedy':
            next_node = probs.argmax(dim=1)
            log_prob = dist.log_prob(next_node)
        elif sampling_type == 'original':
            idx = torch.arange(1 + problem_size, device=device, dtype=torch.long)[None, :].expand_as(probs).clone()
            idx[(probs == 0)] = 1 + problem_size
            next_node = idx.min(dim=1)[0]
            log_prob = dist.log_prob(next_node)

        if selection is not None:
            next_node = selection.reshape(batch_size * beam_size, )
            log_prob = dist.log_prob(next_node)

        next_node = next_node.reshape(batch_size, beam_size, 1)
        log_prob = log_prob.reshape(batch_size, beam_size, 1)
        return next_node, log_prob

    def update_depot_idx(self, prev_global_dones, cur_global_dones, cur_node, step):
        #if idx == 1: return

        where2append = torch.where(
            (prev_global_dones * cur_global_dones) + cur_node.squeeze(2) == 0
        )
        for i in range(where2append[0].size(0)):
            idx = (int(where2append[0][i].item()), int(where2append[1][i].item()))
            self.depot_idx_list[idx[0]][idx[1]].append(step - 1)

        return

    def update_visited_mask(self, cur_node):
        self.visited_mask.scatter_(dim=2, index=cur_node, value=0)
        self.visited_mask[:, :, 0][cur_node.squeeze(2) != 0] = 1
        self.visited_mask[:, :, 0][(self.visited_mask[:, :, 1:] == 0).all(dim=2)] = 1
        return

    def update_capacity_mask(self, ):
        self.capacity_mask = torch.ones(size=(self.batch_size, self.beam_size, 1+self.problem_size), device=self.device)
        self.capacity_mask[self.cur_capacity[:, :, None] - self.depot_node_demand[:, None, :] < 0] = 0
        self.capacity_mask[:, :, 0] = 1
        return

    def update_depot_mask(self, ):
        remaining_capacity = self.total_demand - self.demand_count
        self.depot_mask = torch.ones(size=(self.batch_size, self.beam_size, 1+self.problem_size), device=self.device)
        self.depot_mask[:, :, 0][remaining_capacity > (self.max_vehicle - self.vehicle_count) * self.capacity] = 0
        self.depot_mask[:, :, 0][((self.visited_mask * self.capacity_mask * self.segment_mask)[:, :, 1:] == 0).all(dim=2)] = 1
        return

    def update_demand_state(self, cur_node):
        # cur_node: (batch, beam, 1)
        # depot_node_demand: (batch, graph)

        cur_node_demand = self.depot_node_demand.gather(dim=1, index=cur_node.squeeze(2))
        self.demand_count = self.demand_count + cur_node_demand
        self.cur_capacity = self.cur_capacity - cur_node_demand

        self.cur_capacity[cur_node.squeeze(2) == 0] = self.capacity
        self.vehicle_count = self.vehicle_count + (cur_node.squeeze(2) == 0).to(torch.int)
        return

    def global_dones(self, ):
        return (self.visited_mask[:, :, 1:] == 0).all(dim=2)

    @staticmethod
    def get_cos_sim_matrix(coords):
        coords = coords - coords[:, 0:1, :]
        dot_products = torch.matmul(coords, coords.transpose(1, 2))

        magnitudes = coords.norm(p=2, dim=2, keepdim=True)
        magnitude_matrix = torch.matmul(magnitudes, magnitudes.transpose(1, 2)) + 1e-10

        cos_sim_matrix = dot_products / magnitude_matrix
        return cos_sim_matrix

    @staticmethod
    def get_distance_matrix(coords):
        distances = (coords[:, :, None, :] - coords[:, None, :, :]).norm(p=2, dim=3)
        return distances

    def get_neighbor_mask(self, sim_matrix=None):
        # sim_matrix: (batch, graph, graph)
        if sim_matrix is None: return None

        batch_size, graph_size, _ = sim_matrix.size()

        # (batch, beam, topK)
        _, topk_indices = torch.topk(sim_matrix, k=self.args.topK, dim=2, largest=True)

        neighbor_mask = torch.zeros_like(sim_matrix)
        neighbor_mask.scatter_(dim=2, index=topk_indices, value=1)
        return neighbor_mask


    def gen_pyg_data(self, k_sparse, cvrplib=False):
        norm_demand = self.depot_node_demand / self.capacity

        shift_coords = self.depot_node_xy - self.depot_node_xy[:, 0:1, :]
        _x, _y = shift_coords[:, :, 0], shift_coords[:, :, 1]

        r = torch.sqrt(_x**2 + _y**2)
        theta = torch.atan2(_y, _x)

        depot_node_polar_demand = torch.cat((norm_demand[:, :, None], r[:, :, None], theta[:, :, None]), dim=2)

        if cvrplib:
            cos_mat = (self.cos_sim_matrix + self.cos_sim_matrix.min()) / self.cos_sim_matrix.max()
            euc_mat = 1 - self.euc_dis_matrix

            # (batch, graph, n_sparse)
            topk_values, topk_indices = torch.topk(cos_mat + euc_mat, k=k_sparse, dim=2, largest=True)
            edge_index = topk_indices

            edge_attr1 = cos_mat.gather(dim=2, index=edge_index)
            edge_attr2 = euc_mat.gather(dim=2, index=edge_index)
            edge_attr = torch.cat((edge_attr1[:, :, :, None], edge_attr2[:, :, :, None]), dim=3)

        else:
            # (batch, graph, n_sparse)
            topk_values, topk_indices = torch.topk(self.cos_sim_matrix, k=k_sparse, dim=2, largest=True)
            edge_index = topk_indices

            edge_attr1 = topk_values
            edge_attr2 = self.cos_sim_matrix.gather(dim=2, index=edge_index)
            edge_attr = torch.cat((edge_attr1[:, :, :, None], edge_attr2[:, :, :, None]), dim=3)

        return depot_node_polar_demand, edge_attr, edge_index




