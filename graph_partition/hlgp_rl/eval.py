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

from graph_partition.utils.utils import create_logger
from graph_partition.utils.utils import set_seed
from graph_partition.utils.functions import load_perm_model

from utils import get_subinstances, permute_partition, restore_costs, reorder_instances_by_endpoint
from cvrp_model import GNN
from cvrp_env import OneShotCVRPEnv
from utils import local_partition, select_partition_from_sampling, select_partial_data, select_tail_data
from utils import construct_tail_instances, padding_for_taildata, recover_solution



@torch.no_grad()
def main(args):
    if torch.cuda.is_available() and args.use_cuda:
        args.glb_part_device = torch.device('cuda', index=args.glb_part_device)
        args.loc_part_device = torch.device('cuda', index=args.loc_part_device)
        args.perm_device = torch.device('cuda', index=args.perm_device)
    else:
        args.glb_part_device = args.loc_part_device = args.perm_device = torch.device('cpu')

    create_logger(args, args.log_path, args.perm_model_type, args.problem_size, args.problem_type, args.dist_type)
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

    glb_part_model = GNN(
        units=48,
        node_feats=3,
        k_sparse=args.gnn_topK,
        edge_feats=2,
        depth=args.gnn_depth,
    )
    loc_part_model = GNN(
        units=48,
        node_feats=3,
        k_sparse=args.gnn_topK,
        edge_feats=2,
        depth=args.gnn_depth,
    )

    glb_part_model = glb_part_model.to(args.glb_part_device)
    loc_part_model = loc_part_model.to(args.loc_part_device)

    glb_part_model = load_model(
        glb_part_model, args.glb_model_load_path, args.glb_model_load_epoch, args.glb_part_device, model_type='glb_part'
    )
    loc_part_model = load_model(
        loc_part_model, args.loc_model_load_path, args.loc_model_load_epoch, args.loc_part_device, model_type='loc_part'
    )

    glb_part_model.eval()
    loc_part_model.eval()

    dataset = load_dataset(args.dataset_path, args.valset_size, args.glb_part_device)
    tour_cost, revised_tour_cost, solutions, revised_solutions, duration = evaluation(
        args, revisers, glb_part_model, loc_part_model, dataset
    )

    print('Average Cost: {} +- {}'.format(np.mean(tour_cost), 2 * np.std(tour_cost) / math.sqrt(len(tour_cost))))
    print('Average Cost: {} +- {}'.format(np.mean(revised_tour_cost), 2 * np.std(revised_tour_cost) / math.sqrt(len(revised_tour_cost))))
    print('Duration: {}, {}'.format(duration, duration / len(tour_cost)))

    if args.save_solutions:
        pickle.dump((solutions, revised_solutions), open(os.path.join(args.file_path, 'solutions.pkl'), 'wb'))

    return


def evaluation(args, revisers, glb_part_model, loc_part_model, val_instances):
    start_time = time.time()

    iter_num = int(args.valset_size // args.batch_size)
    assert args.valset_size % args.batch_size == 0
    assert args.valset_size == val_instances[0].size(0)

    tour_cost_list = []
    revised_tour_cost_list = []

    solutions = []
    revised_solutions = []
    for i in range(iter_num):
        st = time.time()
        val_batch = (
            val_instances[0][i * args.batch_size: (i + 1) * args.batch_size],
            val_instances[1][i * args.batch_size: (i + 1) * args.batch_size],
            val_instances[2][i * args.batch_size: (i + 1) * args.batch_size],
            val_instances[3],
        )
        max_vehicle = None
        mask_depot = None
        subp_index = 0

        batch_size = args.batch_size
        problem_size = args.problem_size
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
                args, revisers, subinstances, subdemands, subindices, n_graph_array
            )
            costs = costs.to(device)
            if subp_index == 0:
                tour_cost_list.extend(costs.cpu().numpy().tolist())
                solutions.extend(recover_solution(costs, subinstances, n_graph_array))

            if args.use_local_policy:
                subinstances, subdemands, subindices, cost_list, costs, n_graph_array = \
                    revise_partition(
                        subinstances, subdemands, subindices, cost_list, n_graph_array.reshape((-1, 1)),
                        env.depot_node_xy[:, 0], env.capacity, loc_part_model, revisers, args, revision_iters=args.eval_revision_iters
                    )
                costs = costs.reshape(batch_size, beam_size).to(device)
                n_graph_array = n_graph_array.reshape(batch_size, beam_size)

            if not args.use_subp_eval and subp_index == 0:
                revised_tour_cost_list.extend(costs.cpu().numpy().tolist())
                revised_solutions.extend(recover_solution(costs, subinstances, n_graph_array))
                break

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
                        revise_partition(
                            selected_subinstances, selected_subdemands, selected_subindices, selected_cost_list, selected_ng_array,
                            bkp_depot_node_xy[:, 0], env.capacity, loc_part_model, revisers, args, revision_iters=args.eval_revision_iters
                        )
                    revised_sel_costs = revised_sel_costs.reshape(args.batch_size, 1).to(device)
                else:
                    revised_sel_costs = selected_costs
                    revised_sel_instances = selected_subinstances
                    revised_sel_ng_array = selected_ng_array

                revised_tour_cost_list.extend(revised_sel_costs.cpu().numpy().tolist())
                revised_solutions.extend(recover_solution(revised_sel_costs, revised_sel_instances, revised_sel_ng_array))
                break
            else:
                mask_instance[mask_instance] = valid_instance
                batch_size, problem_size = mask_depot.size()
                subp_index += 1
        print("batch duration: {}".format(time.time() - st))

    duration = time.time() - start_time
    return np.array(tour_cost_list), np.array(revised_tour_cost_list), solutions, revised_solutions, duration





def revise_partition(instances, demands, indices, cost_list, n_graph_array, depot_coord, capacity, loc_part_model, perm_model, args, n_neigh=2, revision_iters=1):
    loc_part_model.eval()
    device = args.loc_part_device

    instances = instances.to(device)
    demands = demands.to(device)
    indices = indices.to(device)
    cost_list = cost_list.to(device)
    depot_coord = depot_coord.to(device)

    for i in range(revision_iters):
        shifts = 0 if i == 0 else 1

        instances, demands, indices, cost_list, n_graph_array = local_partition(
            instances, demands, indices, cost_list, n_graph_array, depot_coord, capacity,
            loc_part_model, perm_model, args, direction=args.revise_direction, shifts=shifts, beam_size=1,
            sampling_type='greedy', use_mode='validation', n_neigh=n_neigh,
            perm_model_type='am_reviser' if i < args.eval_revision_iters - 1 else args.perm_model_type
        )
    costs = restore_costs(cost_list, n_graph_array, args)
    return instances, demands, indices, cost_list, costs, n_graph_array





def load_model(model, path, epoch, device, model_type=''):
    model_index = 'epoch-{}.pth'.format(epoch) if isinstance(epoch, int) else 'best-model.pth'
    if os.path.splitext(path)[-1] == '.pt' or os.path.splitext(path)[-1] == '.pth':
        model_file = path
    else:
        model_file = os.path.join(path, model_index)

    weights = torch.load(model_file, map_location=device)

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
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--problem_size', type=int, default=1000)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--valset_size', type=int, default=128)
    parser.add_argument('--problem_type', type=str, default='cvrp',)

    parser.add_argument('--revision_lens', nargs='+', default=[20], type=int)
    parser.add_argument('--revision_iters', nargs='+', default=[10], type=int)
    parser.add_argument('--gnn_topK', type=int, default=100)
    parser.add_argument('--gnn_depth', type=int, default=12)

    parser.add_argument('--no_aug', action='store_true', default=False)
    parser.add_argument('--no_prune', action='store_true', default=False)
    parser.add_argument('--perm_decode_type', type=str, default='greedy')
    parser.add_argument('--perm_model_type', type=str, default='am_reviser', help='lkh3, am_reviser')

    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--perm_device', type=int, default=0)
    parser.add_argument('--glb_part_device', type=int, default=0)
    parser.add_argument('--loc_part_device', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--dataset_path', type=str, default='')
    parser.add_argument('--glb_model_load_path', type=str, default=None)
    parser.add_argument('--glb_model_load_epoch', type=int, default=None)
    parser.add_argument('--loc_model_load_path', type=str, default=None)
    parser.add_argument('--loc_model_load_epoch', type=int, default=None)
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

    parser.add_argument('--use_subp_eval', type=int, default=0)
    parser.add_argument('--save_solutions', type=int, default=0)

    args = parser.parse_args()

    args.gnn_topK = min(args.problem_size + 1, args.gnn_topK)
    args.perm_model_path = '../../pretrained'
    args.perm_model_stamp = [""]
    args.log_path = os.path.join(args.log_path, 'eval')
    args.use_subp_eval = bool(args.use_subp_eval)
    args.save_solutions = bool(args.save_solutions)
    args.min_max = False

    args.dist_type = 'cross_dist'
    if args.problem_type == 'cvrp' and args.dataset_path == '':
        args.dataset_path = f"../../data/vrp/vrp{args.problem_size}_test_seed1234.pkl"
        args.dist_type = 'uniform'
    main(args)






