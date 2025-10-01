import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data

from typing import Optional

from Bench_Heat.MS_transform.scale_mask import mask_idx, invert_index
from .gnn_layer import MLP


class ParallelTransportPool(gnn.MessagePassing):
    r"""
    Pooling layer with parallel transport

    Args:
        coarse_lvl (int): scale to pool to
        unpool (bool): whether to do unpooling
    """

    def __init__(self, coarse_lvl:int, *, unpool:bool, processor:Optional[str]=None, latent_dim:int):
        super().__init__(aggr="mean", flow="target_to_source")
        self.coarse_lvl = coarse_lvl
        self.unpool = unpool
        self.processor = processor
        self.latent_dim = latent_dim
    
    def _build_block(self, input_dim:int, latent_dim:int):
        
        if self.processor == 'mlp':
            self.block = MLP(input_dim=input_dim, latent_dim=latent_dim, output_dim=latent_dim)
        
        elif self.processor == 'gru':
            # sequence length = 1 trick inside message()
            self.block = nn.GRU(input_size=input_dim, hidden_size=latent_dim, batch_first=True)
        
        elif self.processor == None:
            self.block = nn.Identity()
        
        
    def _apply_block(self, x_in:torch.Tensor):
        self.block = self.block.to(x_in.device)
        if isinstance(self.block, nn.GRU):
            out, _ = self.block(x_in)
            return out                   
        else:
            return self.block(x_in)
    
    def forward(self, data:Data, x:torch.Tensor):
        pool_edge_mask = mask_idx(2 * self.coarse_lvl, data.edge_mask)
        node_idx_fine = torch.nonzero(data.node_mask >= self.coarse_lvl - 1).view(-1)
        node_idx_coarse = torch.nonzero(data.node_mask >= self.coarse_lvl).view(-1)
        node_idx_all_to_fine = invert_index(node_idx_fine, data.num_nodes)
        node_idx_all_to_coarse = invert_index(node_idx_coarse, data.num_nodes)

        coarse, fine = data.edge_index[:, pool_edge_mask]
        coarse_idx_coarse = node_idx_all_to_coarse[coarse]
        fine_idx_fine = node_idx_all_to_fine[fine]

        num_fine, num_coarse = node_idx_fine.shape[0], node_idx_coarse.shape[0]

        if self.unpool:
            edge_index = torch.stack(
                [fine_idx_fine, coarse_idx_coarse]
            )  # Coarse to fine
            size = (num_fine, num_coarse)
        else:  # Pool
            edge_index = torch.stack(
                [coarse_idx_coarse, fine_idx_fine]
            )  # Fine to coarse
            size = (num_coarse, num_fine)
        
        self._build_block(input_dim=x.shape[-1], latent_dim=self.latent_dim)
        out = self.propagate(edge_index=edge_index, x=x, size=size)
        return out

    def message(self, x_j):
        """
        Applies connection to each neighbour, before aggregating for pooling.
        """
        return self._apply_block(x_j)
