import sys

import torch
import time


def coord_transformation(x, valid_size=None):
    valid_size = x.size(1) if valid_size is None else valid_size

    input = x.clone()
    max_x, _ = input[:, :valid_size, 0].max(dim=1)
    max_y, _ = input[:, :valid_size, 1].max(dim=1)
    min_x, _ = input[:, :valid_size, 0].min(dim=1)
    min_y, _ = input[:, :valid_size, 1].min(dim=1)

    diff_x = max_x - min_x
    diff_y = max_y - min_y
    xy_exchanged = diff_y > diff_x

    input[:, :, 0] -= min_x[:, None]
    input[:, :, 1] -= min_y[:, None]

    input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] = input[xy_exchanged, :, 1], input[xy_exchanged, :, 0]
    input /= (torch.max(diff_x, diff_y)[:, None, None] + 1e-10)

    return input



def decompose_func(seed, n_decomp, revision_len, rem_len, shift_len):
    batch_size = seed.size(0)
    seed = torch.roll(seed, dims=1, shifts=-shift_len)

    if rem_len == 0:
        decomp_seed = seed
        rem_seed = None
    else:
        decomp_seed = seed[:, :-rem_len]
        rem_seed = seed[:, -rem_len:]

    if len(decomp_seed.size()) == 3:
        decomp_seed = decomp_seed.reshape(batch_size * n_decomp, revision_len, decomp_seed.size(2))
    elif len(decomp_seed.size()) == 2:
        decomp_seed = decomp_seed.reshape(batch_size * n_decomp, revision_len)

    return decomp_seed, rem_seed






def revision(opts, cost_func, reviser, decomp_seed, decomp_demand, decomp_indice, original_tour, iter=None, embeddings=None):

    revision_len = original_tour.size(0)
    n_graph = decomp_seed.size(0)

    init_cost = cost_func(decomp_seed, original_tour)
    transformed_seeds = coord_transformation(decomp_seed)

    if not opts.no_aug:
        seed2 = torch.cat((1 - transformed_seeds[:, :, [0]], transformed_seeds[:, :, [1]]), dim=2)
        seed3 = torch.cat((transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2)
        seed4 = torch.cat((1 - transformed_seeds[:, :, [0]], 1 - transformed_seeds[:, :, [1]]), dim=2)
        augmented_seeds = torch.cat((transformed_seeds, seed2, seed3, seed4), dim=0)
    else:
        augmented_seeds = transformed_seeds

    if iter == None:
        fwd_cost_revised, fwd_subtour, bwd_cost_revised, bwd_subtour = \
            reviser(augmented_seeds, return_pi=True)
    elif iter == 0:
        fwd_cost_revised, fwd_subtour, bwd_cost_revised, bwd_subtour, embeddings = \
            reviser(augmented_seeds, return_pi=True, return_embedding=True)
    else:
        fwd_cost_revised, fwd_subtour, bwd_cost_revised, bwd_subtour = \
            reviser(augmented_seeds, return_pi=True, embeddings=embeddings)

    if not opts.no_aug:
        _, min_tour_idx = torch.cat([fwd_cost_revised, bwd_cost_revised], dim=0).reshape(8, -1).min(dim=0)
        subtour = torch.cat([fwd_subtour, bwd_subtour], dim=0).reshape(8, -1, revision_len)[min_tour_idx, torch.arange(fwd_subtour.shape[0] // 4), :]

    else:
        _, min_tour_idx = torch.stack((fwd_cost_revised, bwd_cost_revised), dim=0).min(dim=0)
        subtour = torch.stack((fwd_subtour, bwd_subtour), dim=0)[min_tour_idx, torch.arange(n_graph)]

    cost_revised = cost_func(decomp_seed.gather(dim=1, index=subtour.unsqueeze(-1).expand_as(decomp_seed)), subtour)
    subtour[init_cost - cost_revised < 0] = original_tour
    subtour = replace_invalid_subtour(subtour, original_tour, decomp_indice)

    decomp_seed = decomp_seed.gather(dim=1, index=subtour[:, :, None].expand_as(decomp_seed))
    decomp_demand = decomp_demand.gather(dim=1, index=subtour)
    decomp_indice = decomp_indice.gather(dim=1, index=subtour)

    if embeddings is not None:
        if not opts.no_aug:
            embeddings = embeddings.gather(1, subtour.repeat(4, 1).unsqueeze(-1).expand_as(embeddings))
        else:
            embeddings = embeddings.gather(dim=1, index=subtour[:, :, None].expand_as(embeddings))

    return decomp_seed, decomp_demand, decomp_indice, embeddings



def replace_invalid_subtour(subtour, original_tour, indices):
    assert subtour.size() == indices.size()
    assert original_tour.size(0) == indices.size(1) == subtour.size(1)
    new_indices = indices.gather(index=subtour, dim=1)

    is_other = (new_indices != 0)
    is_border = torch.logical_xor(is_other, is_other.roll(dims=1, shifts=1))
    n_border = (is_border.to(torch.int)).sum(dim=1)

    subtour[n_border > 2] = original_tour
    return subtour





def LCP_TSP(
        seed,
        demands,
        indices,
        cost_func,
        reviser,
        revision_len,
        revision_iter,
        opts,
        shift_len,
):
    batch_size, graph_size, _ = seed.shape
    device = seed.device

    reviser.eval()
    reviser.set_decode_type('greedy')

    rem_len = graph_size % revision_len
    n_decomp = int(graph_size // revision_len)
    embeddings = None

    for i in range(revision_iter):
        decomp_seed, rem_seed = decompose_func(seed, n_decomp, revision_len, rem_len, shift_len)
        decomp_demand, rem_demand = decompose_func(demands, n_decomp, revision_len, rem_len, shift_len)
        decomp_indice, rem_indice = decompose_func(indices, n_decomp, revision_len, rem_len, shift_len)

        original_subtour = torch.arange(0, revision_len, dtype=torch.long).to(device)

        if revision_len == graph_size:
            decomp_seed, decomp_demand, decomp_indice, embeddings = revision(
                opts, cost_func, reviser, decomp_seed, decomp_demand, decomp_indice, original_subtour, iter=i, embeddings=embeddings
            )
            embeddings = torch.roll(embeddings, dims=1, shifts=-shift_len)
        else:
            decomp_seed, decomp_demand, decomp_indice, _ = revision(
                opts, cost_func, reviser, decomp_seed, decomp_demand, decomp_indice, original_subtour, iter=None, embeddings=None
            )

        seed = decomp_seed.reshape(batch_size, n_decomp * revision_len, decomp_seed.size(2))
        seed = torch.cat((seed, rem_seed), dim=1) if rem_seed is not None else seed

        demands = decomp_demand.reshape(batch_size, n_decomp * revision_len)
        demands = torch.cat((demands, rem_demand), dim=1) if rem_demand is not None else demands

        indices = decomp_indice.reshape(batch_size, n_decomp * revision_len)
        indices = torch.cat((indices, rem_indice), dim=1) if rem_indice is not None else indices

    return seed, demands, indices





def reconnect(
        get_cost_func,
        seed,
        demands,
        indices,
        opts,
        revisers,
):
    batch_size, graph_size, _ = seed.size()

    if len(revisers) == 0:
        cost_revised = (seed - torch.roll(seed, dims=1, shifts=1)).norm(p=2, dim=2).sum(1)

    for revision_id in range(len(revisers)):
        assert opts.revision_lens[revision_id] <= seed.size(1)
        start_time = time.time()

        shift_len = max(opts.revision_lens[revision_id] // opts.revision_iters[revision_id], 1)
        seed, demands, indices = LCP_TSP(
            seed=seed,
            demands=demands,
            indices=indices,
            cost_func=get_cost_func,
            reviser=revisers[revision_id],
            revision_len=opts.revision_lens[revision_id],
            revision_iter=opts.revision_iters[revision_id],
            opts=opts,
            shift_len=shift_len,
        )

        cost_revised = (seed - torch.roll(seed, dims=1, shifts=1)).norm(p=2, dim=2).sum(1)
        duration = time.time() - start_time


    assert cost_revised.shape == (batch_size, )
    assert seed.shape == (batch_size, graph_size, 2)


    return seed, demands, indices, cost_revised



