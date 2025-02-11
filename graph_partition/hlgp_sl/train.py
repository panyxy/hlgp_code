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
import copy
from copy import deepcopy
from tensorboardX import SummaryWriter

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn


from graph_partition.utils.utils import set_seed
from graph_partition.utils.utils import create_logger
from graph_partition.utils.functions import load_perm_model, load_model

from model.model import BQModel
from utils import get_random_instance, get_distance_matrix
from utils import eval_decoding, reconstruct_tours
from utils import get_subinstances, permute_partition, revise_partition
from utils import prepare_training_instances, recover_solution_from_subindice, prepare_batch_data
from dataset import load_dataset


@torch.no_grad()
def validation(args, part_model, loc_part_model, perm_model, instances_dataset, dataset_info, epoch=None, writer=None, recover_solution=False, use_local_policy=False):
    start_time = time.time()
    part_model.eval()
    loc_part_model.eval()
    perm_model.eval()
    perm_model.set_decode_type('greedy')

    dataset_size, batch_size, problem_size, beam_size = dataset_info
    device = args.glb_part_device

    iter_num = int(dataset_size // batch_size)
    assert dataset_size % batch_size == 0
    assert dataset_size == instances_dataset[0].size(0)

    output_cost_list = []
    solutions = []
    for i in range(iter_num):
        depot_coords, node_coords, node_demands, capacities = (
            instances_dataset[0][i * batch_size: (i + 1) * batch_size],
            instances_dataset[1][i * batch_size: (i + 1) * batch_size],
            instances_dataset[2][i * batch_size: (i + 1) * batch_size],
            instances_dataset[3],
        )
        depot_demands = torch.zeros((batch_size, 1), dtype=torch.int, device=device)

        node_coords = torch.cat((depot_coords, node_coords, depot_coords), dim=1)
        node_demands = torch.cat((depot_demands, node_demands, depot_demands), dim=1)
        capacities = torch.full(size=(batch_size, 1), fill_value=capacities, dtype=torch.int, device=device)
        distance_matrix = get_distance_matrix(node_coords)

        paths, via_depots, tour_lengths = eval_decoding(
            node_coords, node_demands, capacities, distance_matrix, part_model, beam_size, args.knns
        )
        output_cost_list.append(tour_lengths.mean().item())

        if use_local_policy:
            selected_node_list, depot_idx_list = reconstruct_tours(paths, via_depots)
            assert (selected_node_list.sum(dim=2) == (problem_size * (problem_size + 1) / 2)).all()

            subinstances, subdemands, subindices, n_graph_array = get_subinstances(
                node_coords[:, :-1], node_demands[:, :-1], selected_node_list, depot_idx_list
            )
            subinstances, subdemands, subindices, cost_list, costs = permute_partition(
                args, [perm_model], subinstances, subdemands, subindices, n_graph_array
            )
            output_cost_list.append(costs.mean().item())

            subinstances, subdemands, subindices, cost_list, costs, n_graph_array = revise_partition(
                subinstances, subdemands, subindices, cost_list, n_graph_array,
                node_coords[:, 0], instances_dataset[3], loc_part_model, perm_model, args, beam_size
            )
            output_cost_list.append(costs.mean().item())
        if recover_solution:
            solutions.extend(recover_solution_from_subindice(subindices, n_graph_array))

    output_cost_list = np.array(output_cost_list).reshape(-1, 3).sum(axis=0) / iter_num
    if recover_solution:
        print("Epoch: {}, TrainSetCost: {:.5f}, {:.5f}, {:.5f}, Time: {:.2f}".format(epoch, *output_cost_list, time.time() - start_time))
    else:
        print("Epoch: {}, ValCost: {:.5f}, {:.5f}, {:.5f}, Time: {:.2f}\n".format(epoch, *output_cost_list, time.time() - start_time))

    if writer != None and epoch != None:
        writer.add_scalar('Val/Costs_1', output_cost_list[0], epoch)
        writer.add_scalar('Val/Costs_2', output_cost_list[1], epoch)
        writer.add_scalar('Val/Costs_3', output_cost_list[2], epoch)

    return solutions



def main(args):
    if torch.cuda.is_available() and args.use_cuda:
        args.glb_part_device = torch.device('cuda', index=args.glb_part_device)
        args.loc_part_device = torch.device('cuda', index=args.loc_part_device)
        args.perm_device = torch.device('cuda', index=args.perm_device)
    else:
        args.glb_part_device = args.loc_part_device = args.perm_device = torch.device('cpu')

    create_logger(args, args.log_path, args.perm_model_type, args.problem_size, 'cvrp', args.dist_type, part_model_type='bq')
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

    part_model = BQModel(
        dim_input_nodes=4,
        emb_size=args.dim_emb,
        dim_ff=args.dim_ff,
        activation_ff=args.activation_ff,
        nb_layers_encoder=args.nb_layers_encoder,
        nb_heads=args.nb_heads,
        activation_attention=args.activation_attention,
        dropout=args.dropout,
        batchnorm=args.batchnorm,
        problem="cvrp"
    )
    loc_part_model = copy.deepcopy(part_model)
    part_model.transform_coords = True
    loc_part_model.transform_coords = True

    if torch.cuda.device_count() > 1:
        part_model = nn.DataParallel(part_model)
        loc_part_model = nn.DataParallel(loc_part_model)

        args.glb_part_device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
        args.loc_part_device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
        part_model = part_model.to(args.glb_part_device)
        loc_part_model = loc_part_model.to(args.loc_part_device)

        part_model.module.load_state_dict(
            torch.load('../../pretrained/bq_model/cvrp.best', map_location=args.glb_part_device)['net']
        )
        loc_part_model.module.load_state_dict(
            torch.load('../../pretrained/bq_model/cvrp.best', map_location=args.loc_part_device)['net']
        )
        part_optim = torch.optim.Adam(part_model.module.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)
    else:
        part_model = part_model.to(args.glb_part_device)
        loc_part_model = loc_part_model.to(args.loc_part_device)
        part_model.load_state_dict(
            torch.load('../../pretrained/bq_model/cvrp.best', map_location=args.glb_part_device)['net']
        )
        loc_part_model.load_state_dict(
            torch.load('../../pretrained/bq_model/cvrp.best', map_location=args.loc_part_device)['net']
        )
        part_optim = torch.optim.Adam(part_model.parameters(), lr=args.lr, weight_decay=args.l2_weight_decay)

    sl_loss_func = nn.CrossEntropyLoss()
    lr_decay_func = lambda epoch: math.pow(args.lr_decay, int(epoch / args.lr_decay_step))
    part_sched = torch.optim.lr_scheduler.LambdaLR(part_optim, lr_lambda=lr_decay_func)

    starting_epoch = 1
    if args.model_load_enable:
        part_model, part_optim, part_sched, starting_epoch = load_model(
            part_model, part_optim, part_sched, args.model_load_path, args.model_load_epoch,
            args.glb_part_device, model_type='part', load_optim_sched=args.load_optim_sched
        )
    starting_epoch = starting_epoch if args.starting_epoch == -1 else args.starting_epoch

    with torch.autograd.set_detect_anomaly(False):
        train(
            args, part_model, part_optim, part_sched, sl_loss_func, loc_part_model, perm_model, starting_epoch, writer
        )

    return


def train(args, part_model, part_optim, part_sched, sl_loss_func, loc_part_model, perm_model, starting_epoch, writer):
    running_time = 0
    val_instances = get_random_instance(args.valset_size, args.eval_problem_size, 'uniform', args.glb_part_device)

    if starting_epoch == 1 and args.use_eval:
        val_info = (args.valset_size, args.eval_batch_size, args.eval_problem_size, args.eval_beam_size)
        validation(args, part_model, loc_part_model, perm_model, val_instances, val_info,
                   epoch=starting_epoch, writer=writer, use_local_policy=args.use_local_policy)

    for epoch in range(starting_epoch, args.n_epochs + 1):
        start_time = time.time()
        train_epoch(args, part_model, part_optim, sl_loss_func, loc_part_model, perm_model, epoch=epoch, writer=writer)
        part_sched.step()

        duration = time.time() - start_time
        running_time += duration

        if args.use_eval:
            val_info = (args.valset_size, args.eval_batch_size, args.eval_problem_size, args.eval_beam_size)
            validation(args, part_model, loc_part_model, perm_model, val_instances, val_info,
                       epoch=epoch, writer=writer, use_local_policy=args.use_local_policy)

        print('Save checkpoint-{}'.format(epoch))
        checkpoint = {
            'epoch': epoch,
            'part_model_state_dict': part_model.module.state_dict() if isinstance(part_model, nn.DataParallel) else part_model.state_dict(),
            'part_optim_state_dict': part_optim.state_dict(),
            'part_sched_state_dict': part_sched.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.file_path, 'epoch-{}.pth'.format(epoch)))

    print('Training Duration: {}'.format(running_time))

def train_epoch(args, part_model, part_optim, sl_loss_func, loc_part_model, perm_model, epoch, writer=None):
    if not args.train_part_model: return

    print('Epoch: {}, Start TrainingSet Generation'.format(epoch))
    train_instances = get_random_instance(args.trainset_size, args.problem_size, 'uniform', args.glb_part_device)

    train_info = (args.trainset_size, args.eval_batch_size, args.problem_size, args.beam_size)
    solutions = validation(
        args, part_model, loc_part_model, perm_model, train_instances, train_info,
        recover_solution=True, epoch=epoch, use_local_policy=True
    )
    node_coords, node_demands, capacities, remaining_capacities, via_depots = prepare_training_instances(
        train_instances, solutions, reorder_mode='sub_solution'
    )

    training_dataloader = load_dataset(
        node_coords, node_demands, capacities, remaining_capacities, via_depots, args.batch_size, shuffle=True
    )
    print('Epoch: {}, Finish TrainingSet Generation'.format(epoch))

    loss_list = []
    start_time = time.time()
    for update_step in range(args.steps_per_epoch):
        for batch_idx, batch_data in enumerate(training_dataloader):
            loss = train_part_model_with_sl(
                args, part_model, part_optim, sl_loss_func, batch_data, writer=writer
            )
            loss_list.append(loss.item())

    print("Epoch {}, Time: {:.2f}, Loss: {:.5f}".format(epoch, time.time() - start_time, np.mean(loss_list)))
    if writer != None:
        writer.add_scalar('Train/Loss', np.mean(loss_list), epoch)
    return


def train_part_model_with_sl(args, part_model, part_optim, sl_loss_func, batch_data, writer: SummaryWriter=None):
    node_coords, demands, capacities, remaining_capacities, via_depots, _ = prepare_batch_data(batch_data)
    inputs = torch.cat(
        (node_coords,
         (demands / capacities.unsqueeze(-1)).unsqueeze(-1),
         (remaining_capacities / capacities).unsqueeze(-1).repeat(1, node_coords.shape[1]).unsqueeze(-1)), dim=-1
    )
    output_scores = part_model(inputs, demands=demands, remaining_capacities=remaining_capacities)

    ground_truth = torch.full(size=(output_scores.shape[0], ), fill_value=2, dtype=torch.long, device=output_scores.device)
    ground_truth[via_depots[:, 1] == 1.] += 1
    obj_loss = sl_loss_func(output_scores, ground_truth)
    entropy = -torch.log(torch.softmax(output_scores, dim=-1) + 1e-5).mean()
    loss = obj_loss + entropy * args.part_lambda

    part_optim.zero_grad()
    loss.backward()
    if args.use_grad_norm:
        torch.nn.utils.clip_grad_norm_(
            parameters=part_model.parameters(), max_norm=args.max_grad_norm, norm_type=2
        )
    part_optim.step()

    return loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Partition Model for CVRP')
    # RL training environment
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--problem_size', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=20)
    parser.add_argument('--steps_per_epoch', type=int, default=256)
    parser.add_argument('--use_aug', action="store_true", default=False)
    parser.add_argument('--dist_type', type=str, default='uniform')
    parser.add_argument('--trainset_size', type=int, default=100)

    # Network
    parser.add_argument("--dim_emb", type=int, default=192, help="Embeddings size")
    parser.add_argument("--dim_ff", type=int, default=512, help="FF size")
    parser.add_argument("--nb_layers_encoder", type=int, default=9, help="Encoder layers")
    parser.add_argument("--nb_heads", type=int, default=12, help="Number of heads")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout")
    parser.add_argument("--use_batchnorm", dest='batchnorm', action='store_true', help="True = BatchNorm, False = ReZero in encoder")
    parser.add_argument("--activation_ff", type=str, default="relu", help="ReLu or GeLu")
    parser.add_argument("--activation_attention", type=str, default="softmax", help="Softmax or 1.5-entmax")

    # training
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--lr_decay', type=float, default=0.99)
    parser.add_argument('--lr_decay_step', type=float, default=50)
    parser.add_argument('--l2_weight_decay', type=float, default=1e-6, help='l2 penalty in the loss function')
    parser.add_argument('--use_grad_norm', action='store_true', default=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--glb_part_device', type=int, default=2)
    parser.add_argument('--loc_part_device', type=int, default=0)
    parser.add_argument('--perm_device', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_load_enable', action='store_true', default=False)
    parser.add_argument('--model_load_path', type=str, default='')
    parser.add_argument('--model_load_epoch', type=int, default=20)
    parser.add_argument('--log_path', type=str, default='./results')

    # evaluation
    parser.add_argument('--valset_size', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=50)
    parser.add_argument('--eval_beam_size', type=int, default=16)
    parser.add_argument('--eval_problem_size', type=int, default=1000)

    # Permutation Model
    parser.add_argument('--perm_model_type', type=str, default='am_reviser', help='am_reviser, lkh3')
    parser.add_argument('--perm_model_path', type=str, default='../../pretrained')
    parser.add_argument('--perm_model_stamp', nargs='+', default=[""], type=str)
    parser.add_argument('--revision_lens', nargs='+', default=[20], type=int)
    parser.add_argument('--revision_iters', nargs='+', default=[5], type=int)


    parser.add_argument('--starting_epoch', type=int, default=1)
    parser.add_argument('--train_part_model', action='store_true', default=True)
    parser.add_argument('--n_neigh', type=int, default=2)
    parser.add_argument('--revise_direction', type=str, default='forward')

    parser.add_argument('--train_revision_iters', type=int, default=5)
    parser.add_argument('--eval_revision_iters', type=int, default=5)
    parser.add_argument('--use_eval', action='store_true', default=False)
    parser.add_argument('--part_lambda', type=float, default=0.0)

    parser.add_argument('--load_optim_sched', action='store_true', default=False)
    parser.add_argument('--use_local_policy', action='store_true', default=False)
    parser.add_argument('--knns', type=int, default=250)

    args = parser.parse_args()
    args.no_aug = True

    main(args)



