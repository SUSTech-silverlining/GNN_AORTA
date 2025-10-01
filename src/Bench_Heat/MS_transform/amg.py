import torch
import torch_sparse

import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import remove_self_loops, sort_edge_index, coalesce, to_scipy_sparse_matrix

import numpy as np 
from pyamg.classical.split import RS,CLJP

from typing import Tuple

# "Adapted from https://github.com/BGUCompSci/DiffGCNs.py/blob/master/mgpool.py"
def graclus_clustering(edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Performs graph coarsening using the graclus clustering algorithm.
    
    Args:
        edge_index (torch.Tensor)
            
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - cluster (torch.Tensor): Cluster assignment for each node 
            - sorted_perm (torch.Tensor): Sorted permutation indices.
            This represents the mapping from cluster indices to 
            representative node indices in ascending order.
    """
    cluster = gnn.graclus(edge_index)
    cluster, perm = consecutive_cluster(cluster)
    # Sort perm to find the new positions for each index
    sorted_perm, _ = torch.sort(perm)
    return cluster, sorted_perm  # return pool_index and cluster index

def edge_interpolation(cluster:torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    r"""Computes coarsened edge connectivity using cluster-based edge interpolation.
    
    This function creates a new edge index for a coarsened graph by computing 
    P^T A P, where P is the cluster assignment matrix and A is the adjacency 
    matrix. Edges are preserved between clusters based on the original graph 
    connectivity.
    
    Args:
        cluster (torch.Tensor): Cluster assignment for each node 
        edge_index (torch.Tensor): Original graph edges.
        num_nodes (int): Number of nodes in the original graph.
            
    Returns:
        torch.Tensor: Coarsened edge index.
            
    """
    adj_values = torch.ones(edge_index.shape[1])
    cluster, perm = consecutive_cluster(cluster)
    index = torch.stack([cluster, torch.arange(0, num_nodes)], dim=0)
    values = torch.ones(cluster.shape[0], dtype=torch.float)
    uniq, _, _ = torch.unique(cluster, return_inverse=True, return_counts=True)
    newsize = uniq.shape[0]

    origsize = num_nodes

    index, values = torch_sparse.coalesce(
        index, values, m=newsize, n=origsize
    )  # P^T matrix
    new_adj, new_adj_val = torch_sparse.spspmm(
        index,
        values,
        edge_index,
        adj_values,
        m=newsize,
        k=origsize,
        n=origsize,
        coalesced=True,
    )  # P^T A
    index, values = torch_sparse.transpose(
        index, values, m=newsize, n=origsize, coalesced=True
    )  # P
    new_adj, new_adj_val = torch_sparse.spspmm(
        new_adj,
        new_adj_val,
        index,
        values,
        m=newsize,
        k=origsize,
        n=newsize,
        coalesced=True,
    )  # (P^T A) P

    new_adj = remove_self_loops(new_adj)[0]

    # Sort perm to find the new positions for each index
    _ , perm_indices = torch.sort(perm)
    # Create a mapping from original indices to new indices
    index_mapping = torch.zeros_like(perm)
    index_mapping[perm_indices] = torch.arange(len(perm))
    # Map the nodes in new_edge_index using the index_mapping
    mapped_edge_index = index_mapping[new_adj]
    mapped_edge_index = sort_edge_index(mapped_edge_index)

    return mapped_edge_index

# Adapted from https://github.com/baoshiaijhin/amgnet
def RS_clustering(edge_index: torch.Tensor) -> torch.Tensor:
    A = to_scipy_sparse_matrix(edge_index).tocsr()
    
    #splitting=RS(A)
    splitting=RS(A)
    index=np.array(np.nonzero(splitting))
    pool_idx = torch.from_numpy(index)
    pool_idx = torch.squeeze(pool_idx)
    return pool_idx

def PtAP(edge_index:torch.Tensor, index_P:torch.Tensor, num_nodes:int, kN:int):
    r"""come from Ranjan, E., Sanyal, S., Talukdar, P. (2020, April). Asap: Adaptive structure aware pooling
        for learning hierarchical graph representations. AAAI(2020)"""
    
    value = torch.ones(index_P.shape[1])
    
    adj_value = torch.ones(edge_index.shape[1])
    
    # A matrix
    edge_index_A, edge_attr_A = torch_sparse.coalesce(edge_index, adj_value, m=num_nodes, n=num_nodes)
    
    # P^T matrix
    index_P, value_P = torch_sparse.coalesce(index_P, value, m=num_nodes, n=kN)
    
    # P^T A
    index_B, value_B = torch_sparse.spspmm(edge_index_A, edge_attr_A, index_P, value_P, num_nodes, num_nodes, kN)
    
    # P
    index_Pt, value_Pt = torch_sparse.transpose(index_P, value_P, num_nodes, kN)
    
    # (P^T A) P
    index_E, _ = torch_sparse.spspmm(index_Pt, value_Pt, index_B, value_B, kN, num_nodes, kN)

    return index_E


class AMG_Graclus(object):
    def __init__(self, depth: int):

        self.depth = depth

    def __call__(self, data:Data):

        # Create empty tensors to store edge indices and masks
        edge_index = []
        edge_index_local = [data.edge_index]
        edge_mask = []
        node_mask = torch.zeros(data.num_nodes)
        original_idx = torch.arange(data.num_nodes)

        for i in range(self.depth):
            if i == 0:
                pool_idx = original_idx
            else:
                num_nodes = len(pool_idx)
                cluster, pool_idx = graclus_clustering(
                    edge_index=edge_index_local[-1]
                )
                edge = edge_interpolation(cluster, edge_index_local[-1], num_nodes)

            # Update nodes
            original_idx = original_idx[pool_idx]
            node_mask[original_idx] = i  # Mask nodes by level

            if i == 0:
                edge = edge_index_local[-1]
            else:
                edge_index_local.append(edge)
                edge = original_idx[edge]

            edge_index.append(edge)

            edge_mask.append(
                torch.ones(edge.size(1), dtype=torch.long) * (0b1 << (i * 2 + 1))
            )

        edge_index = torch.cat(edge_index, dim=1)
        edge_mask = torch.cat(edge_mask)
        edge_index, edge_mask = coalesce(edge_index, edge_mask, data.num_nodes, "add")

        data.edge_mask = edge_mask
        data.edge_index = edge_index
        data.node_mask = node_mask
        return data

class AMG_RS(object):
    def __init__(self, depth: int):

        self.depth = depth

    def __call__(self, data:Data):

        # Create empty tensors to store edge indices and masks
        edge_index = []
        edge_index_local = [data.edge_index]
        edge_mask = []
        node_mask = torch.zeros(data.num_nodes)
        original_idx = torch.arange(data.num_nodes)

        for i in range(self.depth):
            if i == 0:
                pool_idx = original_idx
            else:
                num_nodes = len(pool_idx)
                pool_idx = RS_clustering(
                    edge_index=edge_index_local[-1]
                )
                kN = pool_idx.size(0)
                perm2 = pool_idx.view(-1, 1)

                # mask contains bool mask of edges which originate from perm (selected) nodes
                mask = (edge_index_local[-1][0]==perm2).sum(0, dtype=torch.bool)
                # create the S
                S0 = edge_index_local[-1][1][mask].view(1, -1)
                S1 = edge_index_local[-1][0][mask].view(1, -1)
                index_P = torch.cat([S0, S1], dim=0)
                
                # relabel for pooling ie: make S [N x kN]
                n_idx = torch.zeros(num_nodes, dtype=torch.long)
                n_idx[pool_idx] = torch.arange(kN)
                index_P[1] = n_idx[index_P[1]]
                edge = PtAP(edge_index_local[-1], index_P, num_nodes, kN)

            # Update nodes
            original_idx = original_idx[pool_idx]
            node_mask[original_idx] = i  # Mask nodes by level

            if i == 0:
                edge = edge_index_local[-1]
            else:
                edge_index_local.append(edge)
                edge = original_idx[edge]

            edge_index.append(edge)

            edge_mask.append(
                torch.ones(edge.size(1), dtype=torch.long) * (0b1 << (i * 2 + 1))
            )

        edge_index = torch.cat(edge_index, dim=1)
        edge_mask = torch.cat(edge_mask)
        edge_index, edge_mask = coalesce(edge_index, edge_mask, data.num_nodes, "add")

        data.edge_mask = edge_mask
        data.edge_index = edge_index
        data.node_mask = node_mask
        return data