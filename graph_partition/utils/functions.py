import pickle

import numpy as np
import random
import torch
import os, sys
from torch.nn import Module
from torch.optim import Optimizer
import json
import torch.nn as nn

from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import pickle

try:
    from node_permutation.AM_Revisier.utils.utils import load_problem
    from node_permutation.AM_Revisier.nets.attention_local import AttentionModel as AttentionModelReviser
except:
    pass

def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_distribution' not in args:
        args['data_distribution'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_distribution'] = dist[0]
    return args



def load_model(model:Module, optimizer:Optimizer, scheduler, path, epoch, device, model_type='', load_optim_sched=False):
    model_index = 'epoch-{}.pth'.format(epoch) if isinstance(epoch, int) else 'best-model.pth'
    model_file = os.path.join(path, model_index)
    weights = torch.load(model_file, map_location=device)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict({**model.module.state_dict(), **weights.get(f'{model_type}_model_state_dict', {})})
    else:
        model.load_state_dict({**model.state_dict(), **weights.get(f'{model_type}_model_state_dict', {})})

    if load_optim_sched:
        optimizer.load_state_dict({**optimizer.state_dict(), **weights.get(f'{model_type}_optim_state_dict', {})})
        scheduler.load_state_dict({**scheduler.state_dict(), **weights.get(f'{model_type}_sched_state_dict', {})})
    epoch = weights.get('epoch', 0) + 1

    return model, optimizer, scheduler, epoch



def load_perm_model(path:str, perm_model_type:str, revision_len:str=None, perm_model_stamp:str=None,
                    epoch:int=None, device=None, decode_type='greedy', ):

    sys.path.append(os.path.join(
        os.path.dirname(os.path.dirname(os.getcwd())), 'node_permutation','AM_Revisier'
    ))

    if perm_model_type == 'am_reviser':
        perm_model_path = os.path.join(
            path, perm_model_type, '{}_{}'.format(perm_model_type, revision_len), perm_model_stamp
        )
    elif perm_model_type == 'lkh3':
        return None
    else:
        raise NotImplementedError

    assert os.path.isdir(perm_model_path)
    args = load_args(os.path.join(perm_model_path, 'args.json'))

    if epoch is None:
        epoch = max(
            int(os.path.splitext(filename)[0].split("-")[1])
            for filename in os.listdir(perm_model_path)
            if os.path.splitext(filename)[1] == '.pt'
        )

    model_index = f'epoch-{epoch}.pt'
    perm_model_file = os.path.join(perm_model_path, model_index)
    assert os.path.isfile(perm_model_file)


    if perm_model_type == 'am_reviser':
        model = AttentionModelReviser(
            args['embedding_dim'],
            args['hidden_dim'],
            load_problem('local'),
            n_encode_layers=args['n_encode_layers'],
            mask_inner=True,
            mask_logits=True,
            normalization=args['normalization'],
            tanh_clipping=args['tanh_clipping'],
            checkpoint_encoder=args.get('checkpoint_encoder', False),
            shrink_size=args.get('shrink_size', None),
        )
    else:
        raise NotImplementedError

    model = model.to(device)
    model_weights = torch.load(perm_model_file, map_location=device)
    model.load_state_dict({**model.state_dict(), **model_weights.get('model', {})})

    model.eval()
    model.set_decode_type(decode_type)

    return model






def run_all_in_pool(func, directory, dataset, opts, use_multiprocessing=True):
    # # Test
    # res = func((directory, 'test', *dataset[0]))
    # return [res]

    assert opts.cpus is not None
    num_cpus = opts.cpus

    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [
                (
                    directory,
                    str(i + offset).zfill(w),
                    *problem
                )
                for i, problem in enumerate(ds)
            ]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus



def vrp2pkl(filepath, filename):
    with open(os.path.join(filepath, filename), 'r') as f:
        dimension = 0
        capacity = 0

        coordinates, demands = [], []

        coord_index = -1
        dem_index = -1
        for line in f:
            line = line.strip('\n')
            line = line.strip('\t')
            if coord_index > -1:
                x, y = line.split('\t')[1:]
                x, y = int(x), int(y)
                coord_index += 1
                coordinates.append([x, y])

                if coord_index == dimension:
                    coord_index = -1

            if dem_index > -1:
                dem = int(line.split('\t')[1])
                dem_index += 1
                demands.append(dem)

                if dem_index == dimension:
                    dem_index = -1

            if line.startswith('DIMENSION'):
                dimension = int(line.split("\t")[-1])

            if line.startswith('CAPACITY'):
                capacity = int(line.split("\t")[-1])

            if line.startswith('NODE_COORD_SECTION'):
                coord_index += 1
                assert dimension != 0 and capacity != 0

            if line.startswith('DEMAND_SECTION'):
                dem_index += 1
                assert dimension != 0 and capacity != 0

        coordinates = np.array(coordinates)
        demands = np.array(demands)

        coordinates[:, 0] = coordinates[:, 0] - coordinates[:, 0].min()
        coordinates[:, 1] = coordinates[:, 1] - coordinates[:, 1].min()

        scale = max(coordinates[:, 0].max(), coordinates[:, 1].max())
        coordinates = coordinates / scale
        data = [[coordinates[0], coordinates[1:], demands[1:], capacity]]
        pickle.dump(data, open(os.path.join(filepath, filename+'.pkl'), 'wb'))

        print("Scale: {}".format(scale))
        print("The data is saved to: {}".format(os.path.join(filepath, filename+'.pkl')))






if __name__ == "__main__":
    pass







