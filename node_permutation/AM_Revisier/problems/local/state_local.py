import torch
from typing import NamedTuple

try:
    from utils.boolmask import mask_long2bool, mask_long_scatter
except:
    from node_permutation.AM_Revisier.utils.boolmask import mask_long2bool, mask_long_scatter



class BsStateLOCAL(NamedTuple):
    loc: torch.Tensor
    dist: torch.Tensor

    batch_ids: torch.Tensor
    beam_ids: torch.Tensor
    last_a: torch.Tensor
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor
    cur_dist: torch.Tensor

    depot_mask: torch.Tensor
    depot_coord: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            raise NotImplementedError

    """
    def __getitem__(self, key):
        raise NotImplementedError
    """

    @staticmethod
    def initialize(loc, depot_mask, depot_coord, visited_dtype=torch.uint8, beam_size=5):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, beam_size, dtype=torch.long, device=loc.device)
        last_a = torch.zeros(batch_size, beam_size, dtype=torch.long, device=loc.device)
        last_a = last_a + 9

        visited_ = (
            torch.zeros(size=(batch_size, beam_size, n_loc), dtype=torch.uint8, device=loc.device)
            if visited_dtype == torch.uint8
            else torch.zeros(batch_size, beam_size, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)
        )

        if depot_mask != None and depot_coord != None:
            depot_mask = depot_mask.to(torch.bool)

            is_depot = ((loc[:, 0:1] - depot_coord).norm(p=2, dim=2) <= 1e-6).expand(batch_size, beam_size)
            visited_[(depot_mask[:, None, :] * is_depot[:, :, None])] = 1

            is_depot = ((loc[:, -1:] - depot_coord).norm(p=2, dim=2) <= 1e-6).expand(batch_size, beam_size)
            visited_[(depot_mask[:, None, :] * is_depot[:, :, None])] = 1

        return BsStateLOCAL(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            batch_ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None].expand(batch_size, beam_size),
            beam_ids=torch.arange(beam_size, dtype=torch.int64, device=loc.device)[None, :].expand(batch_size, beam_size),
            first_a=prev_a,
            last_a=prev_a,
            prev_a=prev_a,
            visited_=visited_,
            lengths=torch.zeros(batch_size, beam_size, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
            depot_mask=depot_mask,
            depot_coord=depot_coord,
            cur_dist=None
        )


    def get_final_cost(self):
        assert self.all_finished()
        batch_size, beam_size = self.first_a.size()

        return self.lengths + (
                self.loc.gather(index=self.first_a[:, :, None].expand(batch_size, beam_size, 2), dim=1) - self.cur_coord
        ).norm(p=2, dim=-1)

    def update(self, selected):
        batch_size, beam_size = selected.size()
        n_loc = self.loc.size(1)

        # (batch, beam)
        prev_a = selected
        # (batch, beam, 2)
        cur_coord = self.loc.gather(index=prev_a[:, :, None].expand(batch_size, beam_size, 2), dim=1)

        cur_dist = self.dist.gather(index=selected[:, :, None].expand(batch_size, beam_size, n_loc), dim=1)

        # (batch, beam)
        lengths = self.lengths
        if self.cur_coord is not None:
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        first_a = prev_a if self.i.item() == 0 else self.first_a

        # visited_: (batch, beam, n_loc)
        # prev_a: (batch, beam)
        if self.visited_.dtype == torch.uint8:
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            raise NotImplementedError

        if self.depot_coord != None and self.depot_mask != None:
            # Mask the repeated depot node
            # (batch, beam)
            is_depot = ((cur_coord - self.depot_coord).norm(p=2, dim=2) <= 1e-6)
            # (batch, beam, n_loc)
            depot_mask = (self.depot_mask[:, None, :] * is_depot[:, :, None])
            visited_[depot_mask] = 1

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_, cur_dist=cur_dist,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0

    def set_state(self, bs_state_ids, ):
        # bs_state_ids: (batch, beam)

        batch_size, beam_size, n_loc = self.visited_.size()

        prev_a = torch.gather(self.prev_a, dim=1, index=bs_state_ids)
        visited_ = torch.gather(self.visited_, dim=1, index=bs_state_ids[:, :, None].expand_as(self.visited_))
        lengths = self.lengths.gather(index=bs_state_ids, dim=1)

        cur_coord = self.cur_coord
        if self.cur_coord is not None:
            cur_coord = self.cur_coord.gather(index=bs_state_ids[:, :, None].expand(batch_size, beam_size, 2), dim=1)

        return self._replace(prev_a=prev_a, visited_=visited_, lengths=lengths, cur_coord=cur_coord,)

    def recover_sequence(self, seleted_node, input, depot_coord, beam_mask, reverse=False):
        # seleced_node: (batch, beam, graph)
        # input: (batch, graph, 2)
        # depot_coord: (batch, 1, 2)
        # beam_mask: (batch, beam)

        batch_size, beam_size, graph_size = seleted_node.size()

        selected_node_list = []
        for i in range(batch_size):
            depot_index = torch.where((depot_coord[i] - input[i]).norm(p=2, dim=1) <= 1e-6)[0]
            #if reverse:
            #    if ((0 - depot_index).abs() == 0).any():
            #        depot_index = depot_index[1:]
            #else:
            #    if ((graph_size - depot_index).abs() == 0).any():
            #        depot_index = depot_index[:-1]

            for j in range(beam_size):
                sequence = seleted_node[i, j]

                if len(depot_index) != 0 and beam_mask[i, j] == False: #0:
                    idx = torch.where(((sequence[:, None] - depot_index[None, :]).abs() <= 1e-6).any(dim=1))[0]

                    n_sel_depot = torch.unique(sequence[idx]).size(0)
                    assert 1 <= n_sel_depot <= 2

                    if n_sel_depot == 1:
                        sequence = torch.cat((sequence[:idx[0]], depot_index, sequence[idx[0]+1:]))[:graph_size]
                    else:
                        assert depot_index[0] == 0 and depot_index[-1] == graph_size-1
                        if reverse:
                            sequence = torch.cat((sequence[:idx[0]], depot_index[1:], sequence[idx[0]+1:]))[:graph_size]
                        else:
                            sequence = torch.cat((sequence[:idx[0]], depot_index[:-1], sequence[idx[0]+1:]))[:graph_size]

                selected_node_list.append(sequence)

        selected_node_list = torch.stack(selected_node_list, dim=0).reshape(batch_size, beam_size, graph_size)
        return selected_node_list


class StateLOCAL(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows
    last_a : torch.Tensor
    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    #depot_mask: torch.Tensor
    #depot_coord: torch.Tensor

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    """
    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
        )
    """


    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        last_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        last_a = last_a + 9

        visited_ = (
            # Visited as mask is easier to understand, as long more memory efficient
            torch.zeros(size=(batch_size, 1, n_loc), dtype=torch.uint8, device=loc.device)
            if visited_dtype == torch.uint8
            else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)
        )

        """
        depot_mask = depot_mask.to(torch.bool)
        is_depot = ((loc[:, 0:1] - depot_coord).norm(p=2, dim=2) <= 1e-6).expand(batch_size, 1)
        visited_[(depot_mask[:, None, :] * is_depot[:, :, None])] = 1

        is_depot = ((loc[:, -1:] - depot_coord).norm(p=2, dim=2) <= 1e-6).expand(batch_size, 1)
        visited_[(depot_mask[:, None, :] * is_depot[:, :, None])] = 1
        """

        return StateLOCAL(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            last_a = prev_a,
            prev_a=prev_a,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=visited_,
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step

        # Add the length
        # cur_coord = self.loc.gather(
        #     1,
        #     selected[:, None, None].expand(selected.size(0), 1, self.loc.size(-1))
        # )[:, 0, :]
        cur_coord = self.loc[self.ids, prev_a]
        lengths = self.lengths
        if self.cur_coord is not None:  # Don't add length for first action (selection of start node)
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Update should only be called with just 1 parallel step, in which case we can check this way if we should update
        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        """
        if self.depot_coord != None and self.depot_mask != None:
            # Mask the repeated depot node
            # (batch, beam)
            is_depot = ((cur_coord - self.depot_coord).norm(p=2, dim=2) <= 1e-6)
            # (batch, beam, n_loc)
            depot_mask = (self.depot_mask[:, None, :] * is_depot[:, :, None])
            visited_[depot_mask] = 1
        """

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_,
                             lengths=lengths, cur_coord=cur_coord, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
            self.dist[
                self.ids,
                self.prev_a
            ] +
            self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
