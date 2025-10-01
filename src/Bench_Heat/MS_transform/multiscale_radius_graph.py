import torch

from torch_sparse import coalesce
from torch_geometric.data import Data
import torch_geometric.nn as gnn


class MultiscaleRadiusGraph:
    r"""Compute a hierarchy of vertex clusters using farthest point sampling and the KNN. For correct
    batch creation, overwrite "torch_geometric.data.Data.__inc__()".


    Edges can belong to multiple levels,
    therefore we store the membership of an edge for a certain level with a bitmask:
        - The bit at position 2 * n corresponds to the edges used for pooling to level n
        - The bit at position 2 * n + 1 corresponds to the edges used for convolution in level n

    To find out if an edge belongs to a level, use a bitwise AND:
        `edge_mask & (0b1 << lvl) > 0`

    Args:
        ratios (list): the ratios for downsampling at each pooling layer.
        radii (list): the radius of the kernel support for each scale.
        max_neighbours (int, optional): the maximum number of neighbors per vertex,
            important to set higher than the expected number of neighbors.
    """

    def __init__(self, ratios, radii, max_neighbours=32):
        assert len(ratios) == len(radii)
        self.ratios = ratios
        self.radii = radii
        self.max_neighbours = max_neighbours

    def __call__(self, data:Data):
        batch = data.batch if "batch" in data else None
        pos = data.pos

        # Create empty tensors to store edge indices and masks
        edge_index = []
        edge_mask = []
        node_mask = torch.zeros(data.num_nodes)

        batch = (
            batch
            if batch is not None
            else torch.zeros(data.num_nodes, dtype=torch.long)
        )

        # Sample points on the surface using farthest point sampling if sample_n is given
        original_idx = torch.arange(data.num_nodes)

        for i, (ratio, radius) in enumerate(zip(self.ratios, self.radii)):
            # POOLING
            if i == 0:
                pool_seeds = original_idx
            else:
                # Sample a subset of vertices
                pool_seeds = gnn.fps(pos, batch, ratio).sort()[0]
                
                # Assigns every point in the original cloud to one of the seed points selected by FPS
                cluster_knn = gnn.knn(
                    x=pos[pool_seeds], y=pos, k=1, batch_x=batch[pool_seeds], batch_y=batch
                )[1]
                
                # # Assign cluster indices to geodesic-nearest vertices via the vector heat method
                # solver = pp3d.PointCloudHeatSolver(pos)  # bless Nicholas Sharp the G.O.A.T.
                # cluster = solver.extend_scalar(pool_seeds.tolist(), np.arange(pool_seeds.numel()))
                # cluster = cluster.round().astype(np.int64)  # round away smoothing

                # fine-coarse mapping
                edge_index.append(
                    torch.stack(
                        (original_idx[pool_seeds][cluster_knn], original_idx), dim=0
                    )
                )
                edge_mask.append(torch.ones_like(original_idx) * (0b1 << (i * 2)))

            # Update nodes
            original_idx = original_idx[pool_seeds]
            pos = pos[pool_seeds]
            batch = batch[pool_seeds]
            node_mask[original_idx] = i  # Mask nodes by level

            if i == 0 and radius == None:
                radius_edges = data.edge_index
            else:
                radius_edges = gnn.radius(pos, pos, radius, batch, batch, self.max_neighbours)

                radius_edges = original_idx[radius_edges]

            edge_index.append(radius_edges)
            edge_mask.append(
                torch.ones(radius_edges.size(1), dtype=torch.long)
                * (0b1 << (i * 2 + 1))
            )

        # Combine and organize edge data
        edge_index = torch.cat(edge_index, dim=1)
        edge_mask = torch.cat(edge_mask)
        edge_index, edge_mask = coalesce(
            edge_index, edge_mask, data.num_nodes, data.num_nodes, "add"
        )

        # Update data object
        data.edge_index = edge_index
        data.edge_mask = edge_mask
        data.node_mask = node_mask
        return data

    def __repr__(self):
        return "{}(radii={}, ratios={})".format(
            self.__class__.__name__, self.radii, self.ratios, self.max_neighbours
        )
