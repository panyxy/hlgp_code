import copy
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import numpy as np
import time
import tqdm
import argparse
import pprint as pp
import random
from typing import List
import pickle
import pickle5
import math
from copy import deepcopy

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F

from graph_partition.utils.utils import set_seed
from graph_partition.utils.utils import create_logger
from graph_partition.utils.functions import load_perm_model, load_model
from utils import get_subinstances, permute_partition, restore_costs
from utils import construct_tail_instances, select_tail_data, padding_for_taildata
from utils import construct_partial_instances, select_partial_data, random_cut_index, subp_cut_index
from cvrp_model import GNN
from cvrp_env import OneShotCVRPEnv, get_random_instance
from utils import local_partition, select_partition_from_sampling
from tensorboardX import SummaryWriter

from graph_partition.hlgp_sl.utils import reconstruct_tours


@torch.no_grad()
def validation(args, glb_part_model, loc_part_model, perm_model, val_instances=None, epoch=None, writer=None):
    glb_part_model.eval()
    loc_part_model.eval()
    perm_model.eval()
    perm_model.set_decode_type('greedy')

    iter_num = int(args.valset_size // args.eval_batch_size)
    assert args.valset_size % args.eval_batch_size == 0
    assert args.valset_size == val_instances[0].size(0)

    output_cost_list = []
    for i in range(iter_num):
        val_batch = (
            val_instances[0][i * args.eval_batch_size: (i + 1) * args.eval_batch_size],
            val_instances[1][i * args.eval_batch_size: (i + 1) * args.eval_batch_size],
            val_instances[2][i * args.eval_batch_size: (i + 1) * args.eval_batch_size],
            val_instances[3],
        )
        max_vehicle = None
        mask_depot = None
        subp_index = 0

        batch_size = args.eval_batch_size
        problem_size = args.eval_problem_size
        beam_size = 1
        sum_solution_index = ((1 + problem_size) * problem_size / 2)
        device = args.glb_part_device

        mask_instance = np.array([True for _ in range(batch_size)])
        base_costs = torch.zeros((batch_size, 1), device=device, dtype=torch.float)
        env_depot_node_indice = torch.arange(problem_size + 1, device=device, dtype=torch.long)[None].repeat(batch_size, 1)

        selected_subinstances = [[] for _ in range(batch_size)]
        selected_subdemands = [[] for _ in range(batch_size)]
        selected_subindices = [[] for _ in range(batch_size)]
        selected_cost_list = [[] for _ in range(batch_size)]

        bkp_data = None
        rem_data = None
        rem_max_length = 0

        while True:
            env = OneShotCVRPEnv(args, batch_size, problem_size, beam_size, device)
            env.load_problems(val_batch)

            reset_state = env.reset(max_vehicle=max_vehicle, mask_depot=mask_depot)
            heatmap = glb_part_model(reset_state)
            _ = env.step(heatmap, sampling_type='greedy')

            if subp_index == 0:
                bkp_depot_node_xy = copy.deepcopy(env.depot_node_xy)
                bkp_depot_node_demand = copy.deepcopy(env.depot_node_demand)

            subinstances, subdemands, subindices, n_graph_array = get_subinstances(
                env.depot_node_xy, env.depot_node_demand, env.selected_node_list, env.depot_idx_list
            )
            subinstances, subdemands, subindices, cost_list, costs = permute_partition(
                args, [perm_model], subinstances, subdemands, subindices, n_graph_array
            )
            costs = costs.to(device)

            if subp_index == 0: output_cost_list.append(costs.mean().item())

            if args.use_local_policy:
                subinstances, subdemands, subindices, cost_list, costs, n_graph_array = \
                    revise_partition(
                        subinstances, subdemands, subindices, cost_list, n_graph_array.reshape((-1, 1)),
                        env.depot_node_xy[:, 0], env.capacity, loc_part_model, perm_model, args
                    )
                costs = costs.reshape(batch_size, beam_size).to(device)
                n_graph_array = n_graph_array.reshape(batch_size, beam_size)

            if subp_index == 0: output_cost_list.append(costs.mean().item())

            if bkp_data == None:
                input_indices, input_cost_list, input_ng_array = subindices, cost_list, n_graph_array
                depot_node_xy, depot_node_demand, depot_node_indice, capacity = env.depot_node_xy, env.depot_node_demand, env_depot_node_indice, env.capacity
            else:
                new_data = (
                    subindices.to(device),
                    cost_list.to(device),
                    n_graph_array,
                    env.depot_node_xy,
                    env.depot_node_demand,
                    env_depot_node_indice,
                    env.capacity,
                )
                input_indices, input_cost_list, input_ng_array, depot_node_xy, depot_node_demand, depot_node_indice, capacity = \
                    select_tail_data(bkp_data, new_data, base_costs, costs, device)

            val_batch, env_depot_node_indice, mask_depot, base_costs, valid_instance, bkp_data, rem_data = construct_tail_instances(
                input_indices, input_cost_list, input_ng_array, depot_node_xy, depot_node_demand, depot_node_indice, capacity, min_max=args.min_max
            )

            rem_idx = 0
            for i in range(args.eval_batch_size):
                if mask_instance[i]:
                    selected_subinstances[i].extend(rem_data[0][rem_idx])
                    selected_subdemands[i].extend(rem_data[1][rem_idx])
                    selected_subindices[i].extend(rem_data[2][rem_idx])
                    selected_cost_list[i].extend(rem_data[3][rem_idx])

                    assert rem_data[0][rem_idx].size(1) == rem_data[1][rem_idx].size(1) == rem_data[2][rem_idx].size(1)
                    rem_max_length = max(rem_max_length, rem_data[0][rem_idx].size(1))
                    rem_idx += 1

            if (valid_instance == False).all():
                assert sum([
                    (torch.cat(sel_subind, dim=0).sum() == sum_solution_index).to(torch.int)
                    for sel_subind in selected_subindices
                ]) == args.eval_batch_size
                selected_cost_list = torch.cat([torch.stack(sel_cost_ls, dim=0) for sel_cost_ls in selected_cost_list], dim=0)

                selected_subinstances, selected_subdemands, selected_subindices, selected_ng_array = \
                    padding_for_taildata(selected_subinstances, selected_subdemands, selected_subindices, bkp_depot_node_xy[:, 0], rem_max_length)
                selected_costs = restore_costs(selected_cost_list, selected_ng_array, args)
                output_cost_list.append(selected_costs.mean().item())

                if args.use_local_policy:
                    revised_sel_instances, revised_sel_demands, revised_sel_indices, revised_sel_cost_list, revised_sel_costs, revised_sel_ng_array = \
                        revise_partition(
                            selected_subinstances, selected_subdemands, selected_subindices, selected_cost_list, selected_ng_array,
                            bkp_depot_node_xy[:, 0], env.capacity, loc_part_model, perm_model, args,
                        )
                    revised_sel_costs = revised_sel_costs.reshape(args.eval_batch_size, 1).to(device)
                else:
                    revised_sel_costs = selected_costs
                output_cost_list.append(revised_sel_costs.mean().item())

                break
            else:
                mask_instance[mask_instance] = valid_instance
                batch_size, problem_size = mask_depot.size()
                subp_index += 1

    output_cost_list = np.array(output_cost_list).reshape(-1, 4).sum(axis=0) / iter_num
    print('\nEpoch-{}: ValCosts:{:.5f}, {:.5f}, {:.5f}, {:.5f}\n'.format(epoch, *output_cost_list))
    if writer != None:
        for i in range(len(output_cost_list)):
            writer.add_scalar('Eval/cost_{}'.format(i), output_cost_list[i], epoch)

    return


def revise_partition(instances, demands, indices, cost_list, n_graph_array, depot_coord, capacity, loc_part_model, perm_model, args, n_neigh=2):
    loc_part_model.eval()
    perm_model.eval()
    perm_model.set_decode_type('greedy')

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
            loc_part_model, perm_model, args, direction=args.revise_direction, shifts=shifts, beam_size=1,
            sampling_type='greedy', use_mode='validation', n_neigh=n_neigh,
        )
    costs = restore_costs(cost_list, n_graph_array, args)
    return instances, demands, indices, cost_list, costs, n_graph_array



def main(args):
    if torch.cuda.is_available() and args.use_cuda:
        args.glb_part_device = torch.device('cuda', index=args.glb_part_device)
        args.loc_part_device = torch.device('cuda', index=args.loc_part_device)
        args.perm_device = torch.device('cuda', index=args.perm_device)
    else:
        args.glb_part_device = args.loc_part_device = args.perm_device = torch.device('cpu')

    create_logger(args, args.log_path, args.perm_model_type, args.problem_size, 'cvrp', args.dist_type)
    pp.pprint(vars(args))
    set_seed(args.seed)
    writer = SummaryWriter(args.file_path)

    if args.perm_model_type == 'am_reviser':
        perm_model = load_perm_model(
            path=args.perm_model_path, perm_model_type=args.perm_model_type, revision_len=args.revision_lens[0],
            perm_model_stamp=args.perm_model_stamp[0], epoch=None, device=args.perm_device,
        )
    else:
        raise NotImplementedError

    glb_part_model = GNN(
        units=args.gnn_embedding_dim,
        node_feats=args.gnn_node_dim,
        k_sparse=args.gnn_topK,
        edge_feats=args.gnn_edge_dim,
        depth=args.gnn_depth,
    )
    loc_part_model = GNN(
        units=args.gnn_embedding_dim,
        node_feats=args.gnn_node_dim,
        k_sparse=args.gnn_topK,
        edge_feats=args.gnn_edge_dim,
        depth=args.gnn_depth,
    )
    glb_part_model = glb_part_model.to(args.glb_part_device)
    loc_part_model = loc_part_model.to(args.loc_part_device)

    glb_part_optim = torch.optim.Adam(glb_part_model.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)
    loc_part_optim = torch.optim.Adam(loc_part_model.parameters(), lr=args.gnn_lr, weight_decay=args.gnn_weight_decay)

    glb_part_sched = torch.optim.lr_scheduler.CosineAnnealingLR(glb_part_optim, T_max=args.n_epochs)
    loc_part_sched = torch.optim.lr_scheduler.CosineAnnealingLR(loc_part_optim, T_max=args.n_epochs)

    starting_epoch = 1
    if args.glb_model_load_enable:
        glb_part_model, glb_part_optim, glb_part_sched, starting_epoch = load_model(
            glb_part_model, glb_part_optim, glb_part_sched, args.glb_model_load_path, args.glb_model_load_epoch,
            args.glb_part_device, model_type='glb_part', load_optim_sched=args.load_glb_optim_sched,
        )
    if args.loc_model_load_enable:
        loc_part_model, loc_part_optim, loc_part_sched, starting_epoch = load_model(
            loc_part_model, loc_part_optim, loc_part_sched, args.loc_model_load_path, args.loc_model_load_epoch,
            args.loc_part_device, model_type='loc_part', load_optim_sched=args.load_loc_optim_sched,
        )
    starting_epoch = starting_epoch if args.starting_epoch == -1 else args.starting_epoch


    with torch.autograd.set_detect_anomaly(False):
        train(
            args, glb_part_model, glb_part_optim, glb_part_sched,
            loc_part_model, loc_part_optim, loc_part_sched, perm_model, starting_epoch, writer
        )

    return



def train(args, glb_part_model, glb_part_optim, glb_part_sched,
          loc_part_model, loc_part_optim, loc_part_sched, perm_model, starting_epoch, writer):
    running_time = 0
    val_instances = get_random_instance(args.valset_size, args.eval_problem_size, 'uniform', args.glb_part_device)

    if starting_epoch == 1 and args.use_eval:
        validation(args, glb_part_model, loc_part_model, perm_model, val_instances, epoch=starting_epoch, writer=writer)

    for epoch in range(starting_epoch, args.n_epochs + 1):
        start_time = time.time()
        train_epoch(
            args, glb_part_model, glb_part_optim, loc_part_model, loc_part_optim, perm_model, epoch=epoch, writer=writer,
        )
        if args.train_global_part: glb_part_sched.step()
        if args.train_local_part: loc_part_sched.step()

        duration = time.time() - start_time
        running_time += duration

        if args.use_eval:
            validation(args, glb_part_model, loc_part_model, perm_model, val_instances, epoch=epoch, writer=writer)

        print('Save checkpoint-{}'.format(epoch))
        checkpoint = {
            'epoch': epoch,
            'glb_part_model_state_dict': glb_part_model.state_dict(),
            'glb_part_optim_state_dict': glb_part_optim.state_dict(),
            'glb_part_sched_state_dict': glb_part_sched.state_dict(),
            'loc_part_model_state_dict': loc_part_model.state_dict(),
            'loc_part_optim_state_dict': loc_part_optim.state_dict(),
            'loc_part_sched_state_dict': loc_part_sched.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.file_path, 'epoch-{}.pth'.format(epoch)))

    print('Training Duration: {}'.format(running_time))
    return


def train_epoch(args, glb_part_model, glb_part_optim, loc_part_model, loc_part_optim, perm_model, epoch=0, writer=None,):

    for update_step in range(args.steps_per_epoch):
        if args.train_global_part:
            train_global_part(
                args, glb_part_model, glb_part_optim, loc_part_model, perm_model,
                epoch=epoch, update_step=update_step, writer=writer
            )

        if args.train_local_part:
            train_local_part(
                args, loc_part_model, loc_part_optim, glb_part_model, perm_model,
                epoch=epoch, update_step=update_step, writer=writer
            )

    return





def train_local_part(args, loc_part_model, loc_part_optim, glb_part_model, perm_model, epoch, update_step, writer=None):
    beam_size = args.loc_beam_size
    device = args.loc_part_device
    revision_part_iters = args.train_loc_revision_iters

    loc_part_model.train()
    glb_part_model.eval()

    subinstances, subdemands, subindices, cost_list, costs, n_graph_array, depot_coord, capacity = \
        execute_glb_part_policy(glb_part_model, perm_model, args, beam_size=1)
    problem_size = 2 * subinstances.size(1)

    subinstances = subinstances.to(device)
    subdemands = subdemands.to(device)
    subindices = subindices.to(device)
    cost_list = cost_list.to(device)
    depot_coord = depot_coord.to(device)

    loss_sum = 0.
    printed_costs = []
    for i in range(revision_part_iters):
        shifts = 0 if i == 0 else 1

        subinstances, subdemands, subindices, cost_list, n_graph_array, loss = local_partition(
            subinstances, subdemands, subindices, cost_list, n_graph_array, depot_coord, capacity,
            loc_part_model, perm_model, args, direction=args.revise_direction, shifts=shifts, beam_size=beam_size,
            sampling_type = 'sampling', use_mode = 'train_local', n_neigh = args.n_neigh,
        )

        loc_part_optim.zero_grad()
        loss.backward()
        if args.use_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                parameters=loc_part_model.parameters(), max_norm=args.max_grad_norm, norm_type=2
            )
        loc_part_optim.step()

        loss_sum += loss.item()

    revised_costs = restore_costs(cost_list, n_graph_array, args)

    printed_costs.extend([costs.mean().item(), revised_costs.mean().item()])
    print("Epoch: %d, Step: %d, Costs: %.5f, %.5f" % (epoch, update_step, *printed_costs))
    if writer != None:
        for i in range(len(printed_costs)):
            writer.add_scalar("Train/local/cost_{}".format(i), printed_costs[i], update_step + epoch * args.steps_per_epoch)

    return


@torch.no_grad()
def execute_glb_part_policy(glb_part_model, perm_model, args, beam_size=None):
    glb_part_model.eval()
    perm_model.eval()
    perm_model.set_decode_type('greedy')

    batch_size = args.loc_batch_size
    problem_size = args.problem_size
    device = args.glb_part_device

    instances = get_random_instance(batch_size, problem_size, 'uniform', device)
    env = OneShotCVRPEnv(args, batch_size, problem_size, beam_size, device)
    env.load_problems(instances)

    reset_state = env.reset()
    heatmap = glb_part_model(reset_state)
    _ = env.step(heatmap, sampling_type='sampling')

    subinstances, subdemands, subindices, n_graph_array = get_subinstances(
        env.depot_node_xy, env.depot_node_demand, env.selected_node_list, env.depot_idx_list
    )
    subinstances, subdemands, subindices, cost_list, costs = permute_partition(
        args, [perm_model], subinstances, subdemands, subindices, n_graph_array
    )
    return subinstances, subdemands, subindices, cost_list, costs, n_graph_array, env.depot_node_xy[:, 0], env.capacity





def train_global_part(args, glb_part_model, glb_part_optim, loc_part_model, perm_model, epoch, update_step, writer=None):
    batch_size = args.batch_size
    beam_size = args.beam_size
    problem_size = args.problem_size
    device = args.glb_part_device
    sum_solution_index = ((1 + problem_size) * problem_size / 2)

    glb_part_model.train()
    loc_part_model.eval()

    instances = get_random_instance(batch_size, problem_size, 'uniform', device)
    max_vehicle = None
    mask_depot = None
    subp_index = 0

    mask_instance = np.array([True for _ in range(args.batch_size)])
    base_costs = torch.zeros((batch_size, 1), device=device, dtype=torch.float)
    env_depot_node_indice = torch.arange(problem_size + 1, device=device, dtype=torch.long)[None].repeat(batch_size, 1)

    selected_subinstances = [[] for _ in range(args.batch_size)]
    selected_subdemands = [[] for _ in range(args.batch_size)]
    selected_subindices = [[] for _ in range(args.batch_size)]
    selected_cost_list = [[] for _ in range(args.batch_size)]

    bkp_data = None
    rem_data = None
    rem_max_length = 0

    printed_costs = []
    no_subp_training = (epoch >= args.no_subp_epoch)
    while True:
        env = OneShotCVRPEnv(args, batch_size, problem_size, beam_size, device)
        env.load_problems(instances)

        reset_state = env.reset(max_vehicle=max_vehicle, mask_depot=mask_depot)
        heatmap = glb_part_model(reset_state)
        log_probs_list = env.step(heatmap, sampling_type='sampling')

        if subp_index == 0:
            bkp_depot_node_xy = copy.deepcopy(env.depot_node_xy)
            bkp_depot_node_demand = copy.deepcopy(env.depot_node_demand)

        subinstances, subdemands, subindices, n_graph_array = get_subinstances(
            env.depot_node_xy, env.depot_node_demand, env.selected_node_list, env.depot_idx_list
        )
        subinstances, subdemands, subindices, cost_list, costs = permute_partition(
            args, [perm_model], subinstances, subdemands, subindices, n_graph_array
        )
        costs = costs.to(device) #- base_costs

        obj_loss = ((costs - costs.mean(dim=1, keepdim=True)) * log_probs_list.sum(dim=2)).mean()
        entropy_loss = -torch.log(heatmap / heatmap.sum(dim=2, keepdim=True)).mean()

        if args.use_local_policy:
            revised_instances, revised_demands, revised_indices, revised_cost_list, revised_costs, revised_ng_array = \
                execute_loc_part_policy(
                    subinstances, subdemands, subindices, cost_list, n_graph_array.reshape((-1, 1)),
                    env.depot_node_xy[:, 0], env.capacity, loc_part_model, perm_model, args,
                )
            revised_costs = revised_costs.reshape(batch_size, beam_size).to(device) #- base_costs
            revised_ng_array = revised_ng_array.reshape(batch_size, beam_size)

            loss = obj_loss + entropy_loss * args.glb_lambda
        else:
            revised_costs = costs
            loss = obj_loss + entropy_loss * args.glb_lambda

        glb_part_optim.zero_grad()
        loss.backward()
        if args.use_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                parameters=glb_part_model.parameters(), max_norm=args.max_grad_norm, norm_type=2
            )
        glb_part_optim.step()

        if subp_index == 0: printed_costs.extend([costs.min(1)[0].mean().item(), revised_costs.min(1)[0].mean().item()])

        if no_subp_training:
            print("Epoch: %d, Step: %d, Costs: %.5f, %.5f" % (epoch, update_step, *printed_costs))
            if writer != None:
                for i in range(len(printed_costs)):
                    writer.add_scalar('Train/global/cost_{}'.format(i), printed_costs[i], update_step + epoch * args.steps_per_epoch)
            break


        if not args.use_local_policy:
            top_instances, top_demands, top_indices, top_cost_list, top_cost, top_ng_array = select_partition_from_sampling(
                subinstances, subdemands, subindices, cost_list, costs, n_graph_array, topK=1
            )
        else:
            top_instances, top_demands, top_indices, top_cost_list, top_cost, top_ng_array = select_partition_from_sampling(
                revised_instances, revised_demands, revised_indices, revised_cost_list, revised_costs, revised_ng_array, topK=1
            )

        if bkp_data == None:
            input_indices, input_cost_list, input_ng_array = top_indices, top_cost_list, top_ng_array
            depot_node_xy, depot_node_demand, depot_node_indice, capacity = env.depot_node_xy, env.depot_node_demand, env_depot_node_indice, env.capacity
        else:
            new_data = (
                top_indices.to(device),
                top_cost_list.to(device),
                top_ng_array,
                env.depot_node_xy,
                env.depot_node_demand,
                env_depot_node_indice,
                env.capacity,
            )
            input_indices, input_cost_list, input_ng_array, depot_node_xy, depot_node_demand, depot_node_indice, capacity = \
                select_tail_data(bkp_data, new_data, base_costs, top_cost, device)

        instances, env_depot_node_indice, mask_depot, base_costs, valid_instance, bkp_data, rem_data = construct_tail_instances(
            input_indices, input_cost_list, input_ng_array, depot_node_xy, depot_node_demand, depot_node_indice, capacity, min_max=args.min_max
        )

        rem_idx = 0
        for i in range(args.batch_size):
            if mask_instance[i]:
                selected_subinstances[i].extend(rem_data[0][rem_idx])
                selected_subdemands[i].extend(rem_data[1][rem_idx])
                selected_subindices[i].extend(rem_data[2][rem_idx])
                selected_cost_list[i].extend(rem_data[3][rem_idx])

                assert rem_data[0][rem_idx].size(1) == rem_data[1][rem_idx].size(1) == rem_data[2][rem_idx].size(1)
                rem_max_length = max(rem_max_length, rem_data[0][rem_idx].size(1))
                rem_idx += 1


        if (valid_instance == False).all():
            assert sum([
                (torch.cat(sel_subind, dim=0).sum() == sum_solution_index).to(torch.int)
                for sel_subind in selected_subindices
            ]) == args.batch_size
            selected_cost_list = torch.cat([torch.stack(sel_cost_ls, dim=0) for sel_cost_ls in selected_cost_list], dim=0)

            selected_subinstances, selected_subdemands, selected_subindices, selected_ng_array = \
                padding_for_taildata(selected_subinstances, selected_subdemands, selected_subindices, bkp_depot_node_xy[:, 0], rem_max_length)
            selected_costs = restore_costs(selected_cost_list, selected_ng_array, args)

            if args.use_local_policy:
                revised_sel_instances, revised_sel_demands, revised_sel_indices, revised_sel_cost_list, revised_sel_costs, revised_sel_ng_array = \
                    execute_loc_part_policy(
                        selected_subinstances, selected_subdemands, selected_subindices, selected_cost_list, selected_ng_array,
                        bkp_depot_node_xy[:, 0], env.capacity, loc_part_model, perm_model, args,
                    )
                revised_sel_costs = revised_sel_costs.reshape(args.batch_size, 1).to(device)
            else:
                revised_sel_costs = selected_costs

            printed_costs.extend([selected_costs.mean().item(), revised_sel_costs.mean().item()])
            print("Epoch: %d, Step: %d, Costs: %.5f, %.5f, %.5f, %.5f" % (epoch, update_step, *printed_costs))
            if writer != None:
                for i in range(len(printed_costs)):
                    writer.add_scalar('Train/global/cost_{}'.format(i), printed_costs[i], update_step + epoch * args.steps_per_epoch)
            break
        else:
            mask_instance[mask_instance] = valid_instance
            batch_size, problem_size = mask_depot.size()
            subp_index += 1

    return


@torch.no_grad()
def execute_loc_part_policy(instances, demands, indices, cost_list, n_graph_array, depot_coord, capacity, loc_part_model, perm_model, args):
    loc_part_model.eval()
    perm_model.eval()
    perm_model.set_decode_type('greedy')

    device = args.loc_part_device
    revision_part_iters = args.train_glb_revision_iters

    instances = instances.to(device)
    demands = demands.to(device)
    indices = indices.to(device)
    cost_list = cost_list.to(device)
    n_graph_array = copy.deepcopy(n_graph_array)
    depot_coord = depot_coord.to(device)

    assert n_graph_array.shape[0] % depot_coord.size(0) == 0.
    depot_coord = depot_coord[:, None, :].expand(depot_coord.size(0), int(n_graph_array.shape[0] // depot_coord.size(0)), 2).reshape(-1, 2)
    assert depot_coord.size(0) == n_graph_array.shape[0]

    for i in range(revision_part_iters):
        shifts = 0 if i == 0 else 1

        instances, demands, indices, cost_list, n_graph_array = local_partition(
            instances, demands, indices, cost_list, n_graph_array, depot_coord, capacity,
            loc_part_model, perm_model, args, direction=args.revise_direction, shifts=shifts, beam_size=1,
            sampling_type='greedy', use_mode='train_global',
        )

    costs = restore_costs(cost_list, n_graph_array, args)
    return instances, demands, indices, cost_list, costs, n_graph_array




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Partition Model for CVRP')
    parser.add_argument('--problem_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=256)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--use_aug', action="store_true", default=False)
    parser.add_argument('--dist_type', type=str, default='uniform')

    parser.add_argument('--gnn_embedding_dim', type=int, default=48)
    parser.add_argument('--gnn_node_dim', type=int, default=3)
    parser.add_argument('--gnn_edge_dim', type=int, default=2)
    parser.add_argument('--gnn_topK', type=int, default=100)
    parser.add_argument('--gnn_depth', type=int, default=12)

    parser.add_argument('--gnn_lr', type=float, default=3e-4)
    parser.add_argument('--gnn_lr_decay', type=float, default=1.0)
    parser.add_argument('--gnn_lr_sched_type', type=str, default='CosAnneal')
    parser.add_argument('--gnn_weight_decay', type=float, default=1e-6, help='l2 penalty in the loss function')
    parser.add_argument('--use_grad_norm', action='store_true', default=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--glb_part_device', type=int, default=0)
    parser.add_argument('--loc_part_device', type=int, default=1)
    parser.add_argument('--perm_device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--glb_model_load_enable', action='store_true', default=False)
    parser.add_argument('--glb_model_load_path', type=str, default=None)
    parser.add_argument('--glb_model_load_epoch', type=int, default=None)
    parser.add_argument('--loc_model_load_enable', action='store_true', default=False)
    parser.add_argument('--loc_model_load_path', type=str, default=None)
    parser.add_argument('--loc_model_load_epoch', type=int, default=None)
    parser.add_argument('--log_path', type=str, default='./results')
    parser.add_argument('--valset_size', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=5)
    parser.add_argument('--eval_problem_size', type=int, default=1000)

    parser.add_argument('--perm_model_type', type=str, default='am_reviser',
                        help='am_reviser, lkh3')
    parser.add_argument('--perm_model_path', type=str, default='../../pretrained')
    parser.add_argument('--perm_model_stamp', nargs='+', default=[""], type=str)
    parser.add_argument('--revision_lens', nargs='+', default=[20], type=int)
    parser.add_argument('--revision_iters', nargs='+', default=[5], type=int)

    parser.add_argument('--starting_epoch', type=int, default=1)
    parser.add_argument('--train_global_part', action='store_true', default=False)
    parser.add_argument('--train_local_part', action='store_true', default=False)
    parser.add_argument('--loc_batch_size', type=int, default=2)
    parser.add_argument('--loc_beam_size', type=int, default=20)
    parser.add_argument('--n_neigh', type=int, default=2)
    parser.add_argument('--revise_direction', type=str, default='forward')

    parser.add_argument('--train_glb_revision_iters', type=int, default=1)
    parser.add_argument('--train_loc_revision_iters', type=int, default=5)
    parser.add_argument('--eval_revision_iters', type=int, default=5)
    parser.add_argument('--use_eval', action='store_true', default=False)
    parser.add_argument('--glb_lambda', type=float, default=0.1)
    parser.add_argument('--loc_lambda', type=float, default=0.005)

    parser.add_argument('--load_glb_optim_sched', action='store_true', default=False)
    parser.add_argument('--load_loc_optim_sched', action='store_true', default=False)
    parser.add_argument('--use_local_policy', action='store_true', default=False)
    parser.add_argument('--no_subp_epoch', type=int, default=2)

    args = parser.parse_args()
    args.no_aug = True
    args.gnn_topK = min(args.problem_size + 1, args.gnn_topK)
    args.min_max = False

    main(args)










