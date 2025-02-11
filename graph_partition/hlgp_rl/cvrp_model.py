import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

from copy import deepcopy
#import torch_geometric.nn as gnn
from torch.nn import BatchNorm1d




class GNN(nn.Module):
    def __init__(self, units, node_feats, k_sparse, edge_feats=1, depth=12):
        super().__init__()
        self.emb_net = GNNEmbNet(depth=depth, units=units, node_feats=node_feats, edge_feats=edge_feats)
        self.part_net = PartNet(units=units, k_sparse=k_sparse)

    def forward(self, reset_state):
        # x: (batch, graph, 3)
        # edge_index: (batch, graph, n_sparse)
        # edge_attr: (batch, graph, n_sparse, 2)
        depot_node_polar_demand = reset_state.depot_node_polar_demand
        edge_index, edge_attr = reset_state.edge_index, reset_state.edge_attr

        emb = self.emb_net(depot_node_polar_demand, edge_index, edge_attr)
        # (batch, graph, n_sparse)
        heu = self.part_net(emb)

        heu = heu / (heu.min(dim=2)[0].min(dim=1)[0][:, None, None] + 1e-5)
        heu = self.reshape(reset_state, heu) + 1e-5
        return heu

    @staticmethod
    def reshape(reset_state, vector):
        batch_size, graph_size, _ = reset_state.depot_node_polar_demand.size()
        device = reset_state.depot_node_polar_demand.device
        matrix = torch.zeros(size=(batch_size, graph_size, graph_size), device=device)
        matrix.scatter_(dim=2, index=reset_state.edge_index, src=vector)
        try:
            assert (matrix.sum(dim=2) >= 0.99).all()
        except:
            torch.save(matrix, './error_reshape.pt')
        return matrix


class GNNEmbNet(nn.Module):
    def __init__(self, depth=12, node_feats=2, edge_feats=1, units=48, act_fn='silu', agg_fn='mean'):  # TODO feats=1
        super().__init__()
        self.depth = depth
        self.node_feats = node_feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        #self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.node_feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        #self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([BatchNorm1d(self.units) for i in range(self.depth)])

        self.e_lin0 = nn.Linear(edge_feats, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        #self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([BatchNorm1d(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        # x: (batch, graph, 3)
        # edge_index: (batch, graph, n_sparse)
        # edge_attr: (batch, graph, n_sparse, 2)
        batch_size, graph_size, n_sparse = edge_index.size()
        n_units = self.units

        x = x
        w = edge_attr

        # (batch, graph, units)
        x = self.v_lin0(x)
        x = self.act_fn(x)
        # (batch, graph, n_sparse, units)
        w = self.e_lin0(w)
        w = self.act_fn(w)

        for i in range(self.depth):
            # (batch, graph, units)
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            # (batch, graph, n_sparse, units)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)

            # x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            # w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))

            x = x0 + self.act_fn(self.v_bns[i](
                (x1 + self.agg_fn(w2, edge_index, x2)).reshape(-1, n_units)
            ).reshape(batch_size, graph_size, n_units))

            w = w0 + self.act_fn(self.e_bns[i](
                (
                    w1 + x3[:, :, None, :].expand(batch_size, graph_size, n_sparse, n_units) + \
                    x4[:, None, :, :].expand(batch_size, graph_size, graph_size, n_units).gather(
                        dim=2, index=edge_index[:, :, :, None].expand(batch_size, graph_size, n_sparse, n_units)
                    )
                ).reshape(-1, n_units)
            ).reshape(batch_size, graph_size, n_sparse, n_units))

        return w

    def agg_fn(self, edge_embed, edge_index, node_embed):
        # edge_embed: (batch, graph, n_sparse, units)
        # edge_index: (batch, graph, n_sparse)
        # node_embed: (batch, graph, units)
        batch_size, graph_size, n_sparse, n_units = edge_embed.size()

        node_embed_expansion = node_embed[:, None, :, :].expand(batch_size, graph_size, graph_size, n_units)
        edge_index_expansion = edge_index[:, :, :, None].expand(batch_size, graph_size, n_sparse, n_units)
        # (batch, graph, n_sparse, n_units)
        incident_node = torch.gather(node_embed_expansion, dim=2, index=edge_index_expansion)

        edge_embed_for_node = (edge_embed * incident_node).mean(dim=2)
        return edge_embed_for_node


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def forward(self, x, k_sparse):
        k_sparse = min(k_sparse, x.size(2))
        for i in range(self.depth):
            # (batch, graph, n_sparse, units)
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                # (batch, graph, n_sparse)
                x = x.reshape(x.size(0), x.size(1), k_sparse)
                x = torch.softmax(x, dim=2)
                # (batch, graph, n_sparse)
                # x = x.flatten()
        return x


# MLP for predicting parameterization theta
class PartNet(MLP):
    def __init__(self, k_sparse, depth=3, units=48, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        self.k_sparse = k_sparse
        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        # (batch, graph, n_sparse)
        return super().forward(x, self.k_sparse).squeeze(dim=-1)











