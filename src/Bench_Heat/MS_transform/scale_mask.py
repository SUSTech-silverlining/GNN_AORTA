import torch
from torch_geometric.data import Data


def invert_index(index, n=None):
    "This function will invert the given global 'index' into the local index"
    n = len(index) if n is None else n
    res = torch.zeros(n, dtype=torch.long, device=index.device)
    res[index] = torch.arange(len(index), device=index.device)
    return res


def mask_idx(lvl, edge_mask):
    """
    Converts a multi-scale edge_mask to a list of indices of edges corresponding to scale lvl.
    :param lvl: the desired scale lvl.
    :param edge_mask: a multi-scale edge mask.
    """
    mask = torch.nonzero(edge_mask & (0b1 << lvl), as_tuple=False).flatten()
    return mask


class ScaleMask:
    r"""Masks the nodes, edges and edge attributes for a given scale level.

    Args:
        lvl (int): the scale level, starting at 0.
    """

    def __init__(self, lvl):
        self.node_lvl = lvl
        self.edge_lvl = (lvl * 2) + 1

    def __call__(self, data:Data):
        mask = mask_idx(
            self.edge_lvl, data.edge_mask
        )  # mask for edges in original graph
        node_idx = torch.nonzero(data.node_mask >= self.node_lvl).view(-1)
        edge_index = data.edge_index[:, mask]
        node_idx_inv = invert_index(node_idx, data.num_nodes)
        edge_index = node_idx_inv[edge_index]

        res = Data(edge_index=edge_index)
        return res


class ScaleMask_bi:
    r"""Masks the nodes, edges and edge attributes for a given scale level.

    suitable for bi_stride, guillard and quadric.

    Args:
        lvl (int): the scale level, starting at 0.
    """

    def __init__(self, lvl):
        self.node_lvl = lvl
        self.edge_lvl = (lvl * 2) + 1

    def __call__(self, data:Data):
        mask = mask_idx(
            self.edge_lvl, data.edge_mask
        )  # mask for edges in original graph
        node_idx = torch.nonzero(data.node_mask >= self.node_lvl).view(-1)
        edge_index = data.edge_index[:, mask]
        res = Data(edge_index=edge_index, batch=data.batch[node_idx])
        return res
