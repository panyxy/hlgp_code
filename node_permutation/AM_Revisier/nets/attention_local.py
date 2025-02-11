import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from torch.nn import DataParallel

try :
    from node_permutation.AM_Revisier.utils.tensor_functions import compute_in_batches
    from node_permutation.AM_Revisier.utils.beamsearch import CachedLookup
    from node_permutation.AM_Revisier.utils.data_utils import sample_many
    from node_permutation.AM_Revisier.nets.graph_encoder import GraphAttentionEncoder
except:
    from utils.tensor_functions import compute_in_batches
    from utils.beamsearch import CachedLookup
    from utils.data_utils import sample_many
    from nets.graph_encoder import GraphAttentionEncoder


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    """
    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key]
            )
    """


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 ):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        # self.attn = LSHAttention(bucket_size=0,n_hashes=4,causal = True)
        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand / prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        else:  # TSP
            # assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )
        
        #self.embedder = Reformer_Encoder(
        #    n_heads=n_heads,
        #    embed_dim=embedding_dim,
        #    n_layers=self.n_encode_layers,
        #    normalization=normalization
        #)

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False, return_embedding=False, embeddings=None, beam_search=False, beam_size=5, beam_temp=1., multi_search=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if isinstance(input, tuple):
            input, depot_mask, depot_coord = input


        if multi_search:
            if embeddings is None:
                embeddings, _ = self.embedder(self._init_embed(input))

            fw_log_probs, fw_selected_node = self._multi_inner(
                input, embeddings, beam_size=beam_size,
            )
            bw_log_probs, bw_selected_node = self._multi_inner(
                input, embeddings, beam_size=beam_size, reverse=True
            )
            bw_selected_node = torch.flip(bw_selected_node, dims=(-1,))

            fw_cost, _ = self.problem.get_bs_costs(input, fw_selected_node)
            bw_cost, _ = self.problem.get_bs_costs(input, bw_selected_node)

            return fw_cost, fw_log_probs, bw_cost, bw_log_probs


        if beam_search:
            if embeddings is None:
                embeddings, _ = self.embedder(self._init_embed(input))

            fw_log_probs, fw_selected_node, fw_beam_mask = self._bs_inner(
                input, embeddings, depot_mask, depot_coord, beam_size=beam_size, beam_temp=beam_temp
            )
            bw_log_probs, bw_selected_node, bw_beam_mask = self._bs_inner(
                input, embeddings, depot_mask, depot_coord, beam_size=beam_size, beam_temp=beam_temp, reverse=True
            )
            bw_selected_node = torch.flip(bw_selected_node, dims=(-1,))

            fw_cost, _ = self.problem.get_bs_costs(input, fw_selected_node)
            bw_cost, _ = self.problem.get_bs_costs(input, bw_selected_node)

            fw_cost[fw_beam_mask] = math.inf
            bw_cost[bw_beam_mask] = math.inf

            if return_embedding:
                return fw_cost, fw_selected_node, bw_cost, bw_selected_node, embeddings
            return fw_cost, fw_selected_node, bw_cost, bw_selected_node

        # return_pi is true means during inference
        if return_pi:
            assert not self.training
        
        # during inference, we make sure x~(0, 1) by coordinate transformation
        # if return_pi:
        #     max_x, indices_max_x = input[:,:,0].max(dim=1)
        #     max_y, indices_max_y = input[:,:,1].max(dim=1)
        #     min_x, indices_min_x = input[:,:,0].min(dim=1)
        #     min_y, indices_min_y = input[:,:,1].min(dim=1)
        #     # shapes: (batch_size, ); (batch_size, )
            
        #     diff_x = max_x - min_x
        #     diff_y = max_y - min_y
        #     xy_exchanged = diff_y > diff_x

        #     # shift to zero
        #     input[:, :, 0] -= (min_x).unsqueeze(-1)
        #     input[:, :, 1] -= (min_y).unsqueeze(-1)

        #     # exchange coordinates for those diff_y > diff_x
        #     input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] =  input[xy_exchanged, :, 1], input[xy_exchanged, :, 0]
            
        #     # scale to (0, 1)
        #     scale_degree = torch.max(diff_x, diff_y)
        #     scale_degree = scale_degree.view(input.shape[0], 1, 1)
        #     input /= scale_degree

        if self.checkpoint_encoder and self.training:
        # The immediate variables are not stored for graident computation.
        # These variables are computed again in the backpropagation.
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        elif embeddings is None:
            embeddings, _ = self.embedder(self._init_embed(input))

        # embeddings shape: (batch size (e.g. width x decomposed pieces), problem size, embedding size)

        # _log_p, pi, entropies = self._inner(input, embeddings)
        # input: (batch, graph, 2)
        # embeddings: (batch, graph, embedding)
        _log_p_fw, pi_fw, _log_p_bw, pi_bw = self._inner(input, embeddings)

        # for inference, inverse coordinate transformation
        # if return_pi:
        #     input[xy_exchanged, :, 0], input[xy_exchanged, :, 1] =  input[xy_exchanged, :, 1], input[xy_exchanged, :, 0]
        #     input *= scale_degree
        #     input[:, :, 0] += (min_x).unsqueeze(-1)
        #     input[:, :, 1] += (min_y).unsqueeze(-1)

        cost_fw, mask_fw = self.problem.get_costs(input, pi_fw)
        cost_bw, mask_bw = self.problem.get_costs(input, pi_bw)
        if return_pi:
            if return_embedding:
                return cost_fw, pi_fw, cost_bw, torch.flip(pi_bw, dims=(-1,)), embeddings
            return cost_fw, pi_fw, cost_bw, torch.flip(pi_bw, dims=(-1,))
            
        ll_fw = self._calc_log_likelihood(_log_p_fw, pi_fw, mask_fw)
        ll_bw = self._calc_log_likelihood(_log_p_bw, pi_bw, mask_bw)
    
        return cost_fw, ll_fw, cost_bw, ll_bw

        
    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        raise NotImplementedError
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        # TSP
        return self.init_embed(input)


    def _multi_inner(self, input, embeddings, beam_size=5, reverse=False):
        batch_size, graph_size, embedding_dim = embeddings.size()
        device = embeddings.device

        log_probs_list = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        seleted_node_list = torch.zeros((batch_size, beam_size, 0), dtype=torch.long, device=device)

        state = self.problem.make_bs_state(input, depot_mask=None, depot_coord=None, beam_size=beam_size, )
        fixed = self._precompute(embeddings)

        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            log_probs, mask = self._get_bs_log_p(fixed, state, i=i, reverse=reverse)

            probs = log_probs.exp().reshape(-1, graph_size)
            dist = torch.distributions.Categorical(probs)

            if self.decode_type == 'sampling':
                next_node = dist.sample()
                log_p = dist.log_prob(next_node)
            elif self.decode_type == 'greedy':
                next_node = probs.argmax(dim=1)
                log_p = dist.log_prob(next_node)

            next_node = next_node.reshape(batch_size, beam_size)
            log_p = log_p.reshape(batch_size, beam_size)

            log_probs_list = log_probs_list + log_p
            seleted_node_list = torch.cat((seleted_node_list, next_node[:, :, None]), dim=2)

            state = state.update(next_node)
            i+= 1

        return log_probs_list, seleted_node_list



    def _bs_inner(self, input, embeddings, depot_mask, depot_coord, beam_size=5, beam_temp=1., reverse=False):
        # depot_mask: (batch, graph)
        # depot_coord: (batch, 2)

        batch_size, graph_size, embedding_dim = embeddings.size()
        device = embeddings.device

        log_probs = torch.zeros((batch_size, beam_size), dtype=torch.float, device=device)
        seleted_node = torch.zeros((batch_size, beam_size, 0), dtype=torch.long, device=device)

        state = self.problem.make_bs_state(input, depot_mask=depot_mask, depot_coord=depot_coord, beam_size=beam_size,)
        fixed = self._precompute(embeddings)
        # node_embeddings: (batch, graph, embedding)
        # context_node_projected: (batch, 1, embedding)
        # glimpse_key: (n_heads, batch, num_steps, graph_size, head_dim)
        # glimpse_val: (n_heads, batch, num_steps, graph_size, head_dim)
        # logit_key: (batch, num_steps, graph_size, embedding)

        bs_valid = torch.full(size=(batch_size, 1), device=device, fill_value=beam_size)
        bs_index = torch.arange(beam_size, device=device)[None, :].expand(batch_size, beam_size)
        beam_mask = torch.where(bs_index < bs_valid.expand(batch_size, beam_size), False, True)
        #beam_mask = torch.where(bs_index < bs_valid.expand(batch_size, beam_size), 0, -math.inf)

        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            log_p, mask = self._get_bs_log_p(fixed, state, i=i, reverse=reverse)

            if i == 1:
                beam_mask[:, 1:] = True #-math.inf

            #bs_reward = -state.lengths + beam_mask
            #seq_probs = torch.softmax(bs_reward / beam_temp, dim=1)[:, :, None]
            #if i != 0:
                #p_sa = (log_probs[:, :, None] + log_p).exp()
                #p_sa = p_sa / p_sa.sum(dim=-1).sum(dim=-1)[:, None, None]

                #p_s = log_probs.exp()
                #p_s = p_s / p_s.sum(dim=-1, keepdim=True)

                #sel_probs = p_sa / (p_s[:, :, None] + 1e-6)
                #sel_probs = sel_probs / (sel_probs.sum(dim=-1, keepdim=True) + 1e-6)
            #else:
            #    sel_probs = log_p.exp()
            #act_probs = (sel_probs * seq_probs).reshape(batch_size, beam_size * graph_size)

            #seq_logit = log_probs + beam_mask
            #seq_probs = torch.softmax(seq_logit / beam_temp, dim=1)[:, :, None]
            #act_probs = (log_p.exp() * seq_probs).reshape(batch_size, beam_size * graph_size)

            act_probs = (log_probs[:, :, None] + log_p).exp()
            act_probs[beam_mask] = 0.
            act_probs = act_probs.reshape(batch_size, beam_size * graph_size)

            top_value, top_index = torch.sort(act_probs, dim=1, descending=True)
            top_value = top_value[:, :beam_size]
            top_index = top_index[:, :beam_size]

            bs_valid = torch.where(top_value == 0, 0, 1).sum(dim=1, keepdim=True)
            beam_mask = torch.where(bs_index < bs_valid.expand(batch_size, beam_size), False, True)

            state_ids = (top_index // graph_size).long()
            action_ids = (top_index % graph_size).long()

            state = state.set_state(state_ids)
            state = state.update(action_ids)

            if i != 0:
                seleted_node = seleted_node.gather(dim=1, index=state_ids[:, :, None].expand_as(seleted_node))
                log_probs = log_probs.gather(dim=1, index=state_ids)

            seleted_node = torch.cat((seleted_node, action_ids[:, :, None]), dim=2)
            log_probs = log_probs + log_p.gather(
                dim=1, index=state_ids[:, :, None].expand_as(log_p)
            ).gather(dim=2, index=action_ids[:, :, None]).squeeze(2)

            i += 1

        #evaluated_log_probs = self.evaluate_probs(input, seleted_node)
        seleted_node = state.recover_sequence(seleted_node, input, depot_coord, beam_mask=beam_mask, reverse=reverse)

        #beam_mask = torch.where(beam_mask == 0., False, True)

        assert (seleted_node.sum(-1)[~beam_mask] == torch.sum(torch.arange(graph_size))).all()
        return log_probs, seleted_node, beam_mask

    def evaluate_probs(self, input, sequences=None):
        # input: (batch, graph, 2)
        # sequences: (batch, 1, graph)

        embeddings, _ = self.embedder(self._init_embed(input))

        batch_size, graph_size, embedding_dim = embeddings.size()
        device = embeddings.device

        state = self.problem.make_state(input)
        fixed = self._precompute(embeddings)

        log_probs = torch.zeros((batch_size, 1), dtype=torch.float, device=device)

        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            # (batch, 1, graph)
            log_p, mask = self._get_log_p(fixed, state, i=i)

            if sequences is None:
                selected = torch.full(size=(batch_size, 1), device=device, dtype=torch.long, fill_value=i)
            else:
                selected = sequences[:, :, i]

            log_probs = log_probs + log_p.gather(dim=2, index=selected[:, :, None]).squeeze(2)

            state = state.update(selected.squeeze())
            i += 1

        return log_probs



    def _inner(self, input, embeddings):

        outputs_fw, sequences_fw = [], []
        outputs_bw, sequences_bw = [], []

        state_fw = self.problem.make_state(input)
        state_bw = self.problem.make_state(input)
        batch_size = state_fw.ids.size(0)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        # node_embeddings: (batch, graph, embedding)
        # context_node_projected: (batch, 1, embedding)
        # glimpse_key: (n_heads, batch, num_steps, graph_size, head_dim)
        # glimpse_val: (n_heads, batch, num_steps, graph_size, head_dim)
        # logit_key: (batch, num_steps, graph_size, embedding)
        fixed = self._precompute(embeddings)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state_fw.all_finished()):
            """
            if self.shrink_size is not None:
                raise NotImplementedError
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished] # TODO double state
                    fixed = fixed[unfinished]
            """

            # log_p, mask,A = self._get_log_p(fixed, state,i=i)
            log_p_fw, mask_fw = self._get_log_p(fixed, state_fw, i=i)
            log_p_bw, mask_bw = self._get_log_p(fixed, state_bw, i=i, reverse=True)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected_fw = self._select_node(log_p_fw.exp()[:, 0, :], mask_fw[:, 0, :])  # Squeeze out steps dimension
            selected_bw = self._select_node(log_p_bw.exp()[:, 0, :], mask_bw[:, 0, :])

            state_fw = state_fw.update(selected_fw)
            state_bw = state_bw.update(selected_bw)

            """
            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                raise NotImplementedError
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)

                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_
            """

            # Collect output of step
            outputs_fw.append(log_p_fw[:, 0, :])
            sequences_fw.append(selected_fw)

            outputs_bw.append(log_p_bw[:, 0, :])
            sequences_bw.append(selected_bw)

            i += 1

        return torch.stack(outputs_fw, 1), torch.stack(sequences_fw, 1), \
               torch.stack(outputs_bw, 1), torch.stack(sequences_bw, 1)


    def sample_many(self, input, batch_rep=1, iter_rep=1,model_local=None):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        raise NotImplementedError
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi,return_two=True),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep,model_local
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        raise NotImplementedError
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )


    def _get_bs_log_p(self, fixed, state, normalize=True, i=0, reverse=False):
        # (batch, beam, embedding)
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_bs_step_context(fixed.node_embeddings, state))

        # glimpse_K: (n_heads, batch, num_steps, graph_size, head_dim)
        # glimpse_V: (n_heads, batch, num_steps, graph_size, head_dim)
        # logit_K: (batch, num_steps, graph_size, embedding)
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        mask = state.get_mask()

        if not reverse:
            if i == 0:
                mask[:, :, :] = True
                mask[:, :, 0] = False

            mask[:, :, -1] = True
            mask[:, :, -1][(mask[:, :, :-1] == True).all(dim=2)] = False

            #if i == mask.size(2) - 1:
            #    mask[:, :, mask.size(2) - 1] = False
        else:
            if i == 0:
                mask[:, :, :] = True
                mask[:, :, -1] = False

            mask[:, :, 0] = True
            mask[:, :, 0][(mask[:, :, 1:] == True).all(dim=2)] = False

            #if i == mask.size(2) - 1:
            #    mask[:, :, 0] = False

        log_p, glimpse = self._bs_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, dist=state.cur_dist)
        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask,



    def _get_log_p(self, fixed, state, normalize=True, i=0, reverse=False):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()
        
        if not reverse:
            if (i == 0):
                # in the first step, only the first node can be selected
                mask[:, :, :] = True
                mask[:, :, 0] = False

            # the end node cannot be selected before the final step
            mask[:, :, mask.shape[2] - 1] = True
            if (i == mask.shape[2] - 1):
                mask[:, :, mask.shape[2] - 1] = False
            # mask[:, :, -1][(mask[:, :, :-1] == True).all(dim=2)] = False
        else:
            if (i == 0):
                # in the first step, only the end node can be selected
                mask[:, :, :] = True
                mask[:, :, mask.shape[2] - 1] = False

            # the starting node cannot be selected before the final step
            mask[:, :, 0] = True
            if (i == mask.shape[2] - 1):
                mask[:, :, 0] = False
            # mask[:, :, 0][(mask[:, :, 1:] == True).all(dim=2)] = False

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask,

    def _get_bs_step_context(self, embeddings, state, from_depot=False):
        current_node = state.get_current_node()
        batch_size, beam_size = current_node.size()

        if self.is_vrp:
            raise NotImplementedError
        else:
            if state.i.item() == 0:
                return self.W_placeholder[None, None, :].expand(batch_size, beam_size, self.W_placeholder.size(-1))
            else:
                fir_node_embeddings = embeddings.gather(
                    dim=1,
                    index=state.first_a[:, :, None].expand(batch_size, beam_size, embeddings.size(-1))
                )
                cur_node_embeddings = embeddings.gather(
                    dim=1,
                    index=current_node[:, :, None].expand(batch_size, beam_size, embeddings.size(-1))
                )
                return torch.cat((fir_node_embeddings, cur_node_embeddings), dim=2)



    
    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps = current_node.size()
  
        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    (
                        state.get_remaining_length()[:, :, None]
                        if self.is_orienteering
                        else state.get_remaining_prize_to_collect()[:, :, None]
                    )
                ),
                -1
            )
        else:  # TSP
            if num_steps == 1:  # We need to special case if we have only 1 step, may be the first or not
                if state.i.item() == 0:
                    # First and only step, ignore prev_a (this is a placeholder)
                    return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
                else:
                    # state.last_a for TSP reviser
                    return embeddings.gather(
                        1,
                        torch.cat((state.last_a, current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                    ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _bs_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, dist=None):
        # query: (batch, beam, embedding)
        # glimpse_K: (n_heads, batch, num_steps, graph_size, head_dim)
        # glimpse_V: (n_heads, batch, num_steps, graph_size, head_dim)
        # logit_K: (batch, num_steps, graph_size, embedding)
        # mask: (batch, beam, graph)
        # dist: (batch, beam, graph)

        glimpse_K = glimpse_K.squeeze(2)
        glimpse_V = glimpse_V.squeeze(2)
        logit_K = logit_K.squeeze(1)

        batch_size, beam_size, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # (n_heads, batch, beam, key_dim)
        glimpse_Q = query.view(batch_size, beam_size, self.n_heads, key_size).permute(2, 0, 1, 3)

        # (n_heads, batch, beam, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))

        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, :].expand_as(compatibility)] = -math.inf

        # (n_heads, batch, beam, head_dim)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # (batch, beam, embedding)
        glimpse = self.project_out(
            heads.permute(1, 2, 0, 3).contiguous().view(batch_size, beam_size, self.n_heads * val_size)
        )
        final_Q = glimpse

        # (batch, beam, graph)
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)) / math.sqrt(final_Q.size(-1))

        if dist is not None:
            score = 1 - logits / logits.max(dim=2)[0][:, :, None]
            score = score - score.mean(dim=2, keepdim=True)
            logits = score + logits

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse






    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)

        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)

        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        
        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):

        if self.is_vrp and self.allow_partial:

            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(state.demands_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
