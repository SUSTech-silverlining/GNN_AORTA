import torch
from torch_cluster import radius_graph
from torch_geometric.nn import fps
import potpourri3d as pp3d
import numpy as np


class HeatSamplingCluster(object):
    """Compute a hierarchy of vertex clusters using farthest point sampling and the vector heat method. For correct
    batch creation, overwrite "torch_geometric.data.Data.__inc__()".

    Args:
        ratios (list): Ratios for farthest point sampling relative to the previous scale hierarchy.
        radii (list): Maximum radii for the creation of radius graphs on each scale.
        loop (bool): Whether to construct self-loop edges.
        max_neighbours (int): Maximum number of neighbours for the radius graphs.
    """

    def __init__(self, ratios, radii, loop=False, max_neighbours=32):
        self.ratios = ratios
        self.radii = radii
        self.loop = loop
        self.max_neighbours = max_neighbours

        self.args = (self.ratios, self.radii, self.loop, self.max_neighbours)

    def __call__(self, data):

        vertices = data.pos
        for i, (ratio, radius) in enumerate(zip(self.ratios, self.radii)):

            if ratio == 1:
                cluster = torch.arange(vertices.shape[0])  # trivial cluster
                edges = radius_graph(vertices, radius, loop=self.loop, max_num_neighbors=self.max_neighbours)
                indices = torch.arange(vertices.shape[0])  # trivial indices

            else:
                # Sample a subset of vertices
                indices = fps(vertices, ratio=ratio)
                indices, _ = indices.sort()  # increases stability

                # Assign cluster indices to geodesic-nearest vertices via the vector heat method
                solver = pp3d.PointCloudHeatSolver(vertices)  # bless Nicholas Sharp the G.O.A.T.
                cluster = solver.extend_scalar(indices.tolist(), np.arange(indices.numel()))
                cluster = cluster.round().astype(np.int64)  # round away smoothing

                # Identify the corresponding vertex subset (discard dropped vertices)
                unique, cluster = torch.unique(torch.from_numpy(cluster), return_inverse=True)
                vertices = vertices[indices[unique]]

                # Connect vertices that are closer together than "radius"
                edges = radius_graph(vertices, radius, loop=self.loop, max_num_neighbors=self.max_neighbours)

                # Indices for scale visualisation and reconstruction
                indices = indices[unique]

            data['scale' + str(i) + '_cluster_map'] = cluster
            data['scale' + str(i) + '_edge_index'] = edges
            data['scale' + str(i) + '_sample_index'] = indices

        return data

    def __repr__(self):
        return '{}(ratios={}, radii={}, loop={}, max_neighbours={})'.format(self.__class__.__name__, *self.args)
