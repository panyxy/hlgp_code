"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import numpy as np
import torch
from torch.nn import Module, Linear, Embedding
from torch.nn.modules import ModuleList
try:
    from encoder import EncoderLayer
except:
    from graph_partition.hlgp_sl.model.encoder import EncoderLayer




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




class BQModel(Module):

    def __init__(self, dim_input_nodes, emb_size, dim_ff, activation_ff, nb_layers_encoder, nb_heads,
                 activation_attention, dropout, batchnorm, problem="tsp", transform_coords=False):
        assert problem == "tsp" or problem == "cvrp" or problem == "kp" or problem == "op"
        super().__init__()
        self.problem = problem
        self.input_emb = Linear(dim_input_nodes, emb_size)
        if problem != "kp":
            self.begin_end_tokens = Embedding(num_embeddings=2, embedding_dim=emb_size)

        self.nb_layers_encoder = nb_layers_encoder
        self.encoder = ModuleList([EncoderLayer(nb_heads, activation_attention, emb_size, dim_ff, activation_ff,
                                                dropout, batchnorm) for _ in range(nb_layers_encoder)])

        if problem == "cvrp":
            output_dim = 2
        else:
            output_dim = 1
        self.scores_projection = Linear(emb_size, output_dim)
        self.transform_coords = transform_coords

    def forward(self, inputs, **problem_data):
        if self.problem == "cvrp" and self.transform_coords:
            coords, others = torch.split(inputs, (2, 2), dim=-1)
            coords = coord_transformation(coords)
            inputs = torch.cat((coords, others), dim=-1)

        # inputs
        #     TSP [batch_size, seq_len, (x_coord, y_coord)]
        #    CVRP [batch_size, seq_len, (x_coord, y_coord, demand, current_capacity)]
        #      OP [batch_size, seq_len, (x_coord, y_coord, node_value, upper_bound)]
        #      KP [batch_size, seq_len, (weight, value, remaining_capacity)]
        if self.problem == "op":
            assert "dist_matrices" in problem_data

        # [batch_size, seq_len, emb_size]
        input_emb = self.input_emb(inputs)
        if self.problem != "kp":
            # [batch_size, emb_size]
            input_emb[:, 0, :] += self.begin_end_tokens(torch.tensor([[0]], device=inputs.device)).squeeze(1)
            input_emb[:, -1, :] += self.begin_end_tokens(torch.tensor([[1]], device=inputs.device)).squeeze(1)

        state = input_emb
        for layer in self.encoder:
            # [batch_size, seq_len, emb_size]
            state = layer(state)

        # [batch_size, seq_len] or [batch_size, seq_len, 2]
        scores = self.scores_projection(state).squeeze(-1)

        if self.problem == "tsp":
            # mask origin and destination
            scores[:, 0] = scores[:, -1] = -np.inf
        elif self.problem == "cvrp":
            # the 1st logit denotes the probability of selecing this node
            # the 2nd logit denotes the probability of the node is from the depot
            # mask origin and destination (x2 - direct edge and via depot)
            scores[:, 0, :] = scores[:, -1, :] = -np.inf
            # exclude all impossible edges (direct edges to nodes with capacity larger than available demand)
            scores[..., 0][problem_data["demands"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf

            mask_depot = problem_data.get('mask_depot', None)
            if mask_depot != None:
                assert mask_depot.size() == (scores.size(0), scores.size(1))
                scores[:, :, 1][mask_depot.bool()] = -np.inf

                zero_demand_idx = torch.cat((
                    torch.zeros((1, ), device=scores.device, dtype=torch.int),
                    (problem_data["demands"][:, 1:-1] == 0).to(torch.int).sum(dim=1)
                ), dim=0).cumsum(dim=0)[:-1]
                first_depot_idx = torch.where(problem_data["demands"][:, 1:-1] == 0)[1][zero_demand_idx] + 1

                non_customer_batch = (mask_depot[:, 1:-1] == 1).all(dim=1)
                first_depot_idx = first_depot_idx[non_customer_batch]
                mask_depot[non_customer_batch, first_depot_idx] = 0
                scores[:, :, 0][mask_depot.bool()] = -np.inf


        elif self.problem == "op":
            scores[:, 0] = scores[:, -1] = -np.inf
            # op - mask all nodes with cost to go there and back to depot > current upperbound
            # todo: update with real values
            scores[problem_data["dist_matrices"][:, 0] +
                   problem_data["dist_matrices"][:, -1] - inputs[..., 3] > 1e-6] = -np.inf
        elif self.problem == "kp":
            # kp - mask all nodes with weights > current capacity
            scores[problem_data["weights"] > problem_data["remaining_capacities"].unsqueeze(-1)] = -np.inf

        return scores.reshape(scores.shape[0], -1)
