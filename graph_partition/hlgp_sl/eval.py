import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import argparse
import copy
import numpy as np

import torch
import pprint as pp
import time
import random
import math
import pickle
import pickle5

import torch.nn.functional as F
import torch.nn as nn

from graph_partition.utils.utils import create_logger
from graph_partition.utils.utils import set_seed
from graph_partition.utils.functions import load_perm_model
from model.model import BQModel

from utils import get_subinstances, permute_partition, revise_partition
from utils import eval_decoding, reconstruct_tours, get_distance_matrix, recover_solution



@torch.no_grad()
def main(args):
    if torch.cuda.is_available() and args.use_cuda:
        args.glb_part_device = torch.device('cuda', index=args.glb_part_device)
        args.loc_part_device = torch.device('cuda', index=args.loc_part_device)
        args.perm_device = torch.device('cuda', index=args.perm_device)
    else:
        args.glb_part_device = args.loc_part_device = args.perm_device = torch.device('cpu')

    create_logger(args, args.log_path, args.perm_model_type, args.problem_size, args.problem_type, args.dist_type, part_model_type='bq')
    pp.pprint(vars(args))
    set_seed(args.seed)

    revisers = []
    revision_lens = args.revision_lens

    for reviser_size in revision_lens:
        reviser = load_perm_model(
            path=args.perm_model_path, perm_model_type='am_reviser', revision_len=reviser_size,
            perm_model_stamp=args.perm_model_stamp[0], epoch=None, device=args.perm_device,
            decode_type=args.perm_decode_type
        )
        revisers.append(reviser)

    part_model = BQModel(
        dim_input_nodes=4,
        emb_size=192,
        dim_ff=512,
        activation_ff='relu',
        nb_layers_encoder=9,
        nb_heads=12,
        activation_attention='softmax',
        dropout=0.0,
        batchnorm=False,
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
    else:
        part_model = part_model.to(args.glb_part_device)
        loc_part_model = loc_part_model.to(args.loc_part_device)
        part_model.load_state_dict(
            torch.load('../../pretrained/bq_model/cvrp.best', map_location=args.glb_part_device)['net']
        )
        loc_part_model.load_state_dict(
            torch.load('../../pretrained/bq_model/cvrp.best', map_location=args.loc_part_device)['net']
        )

    if args.model_load_path != "":
        part_model = load_model(
            part_model, args.model_load_path, args.model_load_epoch, args.glb_part_device, model_type='part'
        )
    part_model.eval()
    loc_part_model.eval()


    dataset = load_dataset(args.dataset_path, args.valset_size, args.glb_part_device)
    tour_cost, revised_tour_cost, solutions, revised_solutions, duration = evaluation(args, revisers, part_model, loc_part_model, dataset)

    print('Average Cost: {} +- {}'.format(np.mean(tour_cost), 2 * np.std(tour_cost) / math.sqrt(len(tour_cost))))
    print('Average Cost: {} +- {}'.format(np.mean(revised_tour_cost), 2 * np.std(revised_tour_cost) / math.sqrt(len(revised_tour_cost))))
    print('Duration: {}, {}'.format(duration, duration / len(tour_cost)))

    if args.save_solutions:
        pickle.dump((solutions, revised_solutions), open(os.path.join(args.file_path, 'solutions.pkl'), 'wb'))

    return



def evaluation(args, revisers, part_model, loc_part_model, val_instances):
    start_time = time.time()

    iter_num = int(args.valset_size // args.batch_size)
    assert args.valset_size % args.batch_size == 0
    assert args.valset_size == val_instances[0].size(0)

    batch_size = args.batch_size
    problem_size = args.problem_size
    beam_size = args.beam_size
    device = args.glb_part_device

    tour_cost_list = []
    revised_tour_cost_list = []

    solutions = []
    revised_solutions = []
    for i in range(iter_num):
        depot_coords, node_coords, node_demands, capacities = (
            val_instances[0][i * batch_size: (i + 1) * batch_size],
            val_instances[1][i * batch_size: (i + 1) * batch_size],
            val_instances[2][i * batch_size: (i + 1) * batch_size],
            val_instances[3],
        )
        depot_demands = torch.zeros((batch_size, 1), dtype=torch.int, device=device)

        node_coords = torch.cat((depot_coords, node_coords, depot_coords), dim=1)
        node_demands = torch.cat((depot_demands, node_demands, depot_demands), dim=1)
        capacities = torch.full(size=(batch_size, 1), fill_value=capacities, dtype=torch.int, device=device)
        distance_matrix = get_distance_matrix(node_coords)

        paths, via_depots, tour_lengths = eval_decoding(
            node_coords, node_demands, capacities, distance_matrix, part_model, beam_size, args.knns
        )
        selected_node_list, depot_idx_list = reconstruct_tours(paths, via_depots)
        assert (selected_node_list.sum(dim=2) == (problem_size * (problem_size + 1) / 2)).all()

        subinstances, subdemands, subindices, n_graph_array = get_subinstances(
            node_coords[:, :-1], node_demands[:, :-1], selected_node_list, depot_idx_list
        )
        subinstances, subdemands, subindices, cost_list, costs = permute_partition(
            args, revisers, subinstances, subdemands, subindices, n_graph_array
        )
        tour_cost_list.extend(costs.cpu().numpy().tolist())
        solutions.extend(recover_solution(costs, subinstances, n_graph_array))

        if args.use_local_policy:
            subinstances, subdemands, subindices, cost_list, costs, n_graph_array = revise_partition(
                subinstances, subdemands, subindices, cost_list, n_graph_array,
                node_coords[:, 0], val_instances[3], loc_part_model, revisers, args, args.beam_size
            )
        revised_tour_cost_list.extend(costs.cpu().numpy().tolist())
        revised_solutions.extend(recover_solution(costs, subinstances, n_graph_array))

        print("batch results: ", np.mean(tour_cost_list), np.mean(revised_tour_cost_list))

    duration = time.time() - start_time
    return np.array(tour_cost_list), np.array(revised_tour_cost_list), solutions, revised_solutions, duration



def load_model(model, path, epoch, device, model_type=''):
    model_index = 'epoch-{}.pth'.format(epoch) if isinstance(epoch, int) else 'best-model.pth'
    if os.path.splitext(path)[-1] == '.pt' or os.path.splitext(path)[-1] == '.pth':
        model_file = path
    else:
        model_file = os.path.join(path, model_index)

    weights = torch.load(model_file, map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict({**model.module.state_dict(), **weights.get(f'{model_type}_model_state_dict', {})})
    else:
        model.load_state_dict({**model.state_dict(), **weights.get(f'{model_type}_model_state_dict', {})})
    return model

def load_dataset(dataset_path, valset_size, device):
    with open(dataset_path, 'rb') as f:
        try:
            data = pickle.load(f)
        except:
            data = pickle5.load(f)
    data = data[:valset_size]

    depot_coord, node_coord, node_demand = [], [], []
    for instance in data:
        depot_coord.append(instance[0])
        node_coord.append(instance[1])
        node_demand.append(instance[2])

    depot_coord = torch.tensor(depot_coord, dtype=torch.float, device=device)[:, None]
    node_coord = torch.tensor(node_coord, dtype=torch.float, device=device)
    node_demand = torch.tensor(node_demand, dtype=torch.int, device=device)

    return (depot_coord, node_coord, node_demand, int(data[0][-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--problem_size', type=int, default=1000)
    parser.add_argument('--beam_size', type=int, default=32)
    parser.add_argument('--valset_size', type=int, default=100)
    parser.add_argument('--problem_type', type=str, default='cvrp', help='cvrp')

    parser.add_argument('--revision_lens', nargs='+', default=[20], type=int)
    parser.add_argument('--revision_iters', nargs='+', default=[10], type=int)
    parser.add_argument('--knns', type=int, default=250)

    parser.add_argument('--no_aug', action='store_true', default=False)
    parser.add_argument('--no_prune', action='store_true', default=False)
    parser.add_argument('--perm_decode_type', type=str, default='greedy')
    parser.add_argument('--perm_model_type', type=str, default='am_reviser', help='lkh3, am_reviser')

    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--perm_device', type=int, default=0)
    parser.add_argument('--glb_part_device', type=int, default=0)
    parser.add_argument('--loc_part_device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--model_load_path', type=str, default=None)
    parser.add_argument('--model_load_epoch', type=int, default=None)
    parser.add_argument('--log_path', type=str, default='./results')

    parser.add_argument('--revise_direction', type=str, default='forward')
    parser.add_argument('--eval_revision_iters', type=int, default=5)
    parser.add_argument('--use_local_policy', action='store_true', default=False)

    parser.add_argument('--cpus', type=int, default=32, help="Number of CPUs to use")
    parser.add_argument('--disable_cache', action='store_true', default=True, help='Disable caching')
    parser.add_argument('--max_calc_batch_size', type=int, default=1000, help='Size for subbatches')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    parser.add_argument('-n', type=int, help="Number of instances to process")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--save_solutions', type=int, default=0)


    args = parser.parse_args()
    args.save_solutions = bool(args.save_solutions)

    args.perm_model_path = '../../pretrained'
    args.perm_model_stamp = [""]
    args.log_path = os.path.join(args.log_path, 'eval')

    args.dist_type = 'cross_dist'
    if args.problem_type == 'cvrp' and args.dataset_path == '':
        args.dataset_path = f"../../data/vrp/vrp{args.problem_size}_test_seed1234.pkl"
        args.dist_type = 'uniform'

    main(args)
