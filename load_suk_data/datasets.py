import torch_geometric as pyg
import torch
import os
from glob import glob
import h5py
from tqdm import tqdm
from torch_geometric.data import Data
from pathlib import Path


class Dataset(pyg.data.Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        self.path_to_hdf5_file = sorted(glob(os.path.join(self.root, "raw", "*.hdf5")))[0]

        with h5py.File(self.path_to_hdf5_file, 'r') as hdf5_file:
            sample_ids = [os.path.join(self.path_to_hdf5_file, sample_id) for sample_id in hdf5_file]

        return [os.path.relpath(sample_id, os.path.join(self.root, "raw")) for sample_id in sample_ids]

    @property
    def processed_file_names(self):
        return [f"data_{idx}.pt" for idx in range(len(self.raw_file_names))]

    def download(self):
        return

    def process(self):
        for idx, path in enumerate(tqdm(self.raw_paths, desc="Reading & transforming", leave=False)):
            data = self.read_hdf5_data(path)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, self.processed_file_names[idx]))

    # @staticmethod
    # def read_hdf5_data(path):
    #     path_to_hdf5_file, sample_id = os.path.split(path)

    #     with h5py.File(path_to_hdf5_file, 'r') as hdf5_file:

    #         data = Data(
    #             y=torch.from_numpy(hdf5_file[sample_id]['velocity'][()]),
    #             pos=torch.from_numpy(hdf5_file[sample_id]['pos_tets'][()]),
    #             tets=torch.from_numpy(hdf5_file[sample_id]['tets'][()].T),
    #             inlet_index=torch.from_numpy(hdf5_file[sample_id]['inlet_idcs'][()]),
    #             lumen_wall_index=torch.from_numpy(hdf5_file[sample_id]['lumen_wall_idcs'][()]),
    #             outlets_index=torch.from_numpy(hdf5_file[sample_id]['outlets_idcs'][()])
    #         )

    #     return data

    @staticmethod
    def read_hdf5_data(path):
        """
        Read a single sample from HDF5 as PyG Data:
        - surface/WSS (single): y=[pressure, wss(3)], pos, face
        - volume/velocity (bifurcating): y=velocity or [pressure, velocity(3)], pos, tets
        - Automatically attach available boundary condition indices inlet/lumen_wall/outlets
        """

        def _first_key(g, candidates, required=True):
            for k in candidates:
                if k in g:
                    return k
            if required:
                raise KeyError(f"None of {candidates} found. Available keys: {list(g.keys())}")
            return None

        path_to_hdf5_file, sample_id = os.path.split(path)
        with h5py.File(path_to_hdf5_file, 'r') as f:
            g = f[sample_id]
            keys = set(g.keys())

            # ======== Determine task type ========
            is_surface = any(k in keys for k in ['wss', 'wss_vec', 'wall_shear_stress'])
            is_volume  = any(k in keys for k in ['velocity', 'u', 'vel'])

            if is_surface:
                # ---- surface / WSS: requires wss + pressure ----
                wss_key = _first_key(g, ['wss', 'wss_vec', 'wall_shear_stress'])
                pos_key = _first_key(g, ['pos', 'points', 'vertices'])
                face_key = _first_key(g, ['face', 'faces', 'triangles', 'tris', 'cells'])
                pres_key = _first_key(g, ['pressure', 'p'])  # must have pressure

                pos = torch.as_tensor(g[pos_key][()], dtype=torch.float32)           # [N,3]
                wss = torch.as_tensor(g[wss_key][()], dtype=torch.float32)           # [N,3] (usually)
                pressure = torch.as_tensor(g[pres_key][()], dtype=torch.float32)     # [N] / [N,1]
                if pressure.ndim == 1:
                    pressure = pressure.view(-1, 1)
                if wss.ndim == 1:  # safety check
                    wss = wss.view(-1, 1)

                y = torch.cat([pressure, wss], dim=1)                                # [N,4]

                faces_np = g[face_key][()]
                faces = torch.as_tensor(faces_np, dtype=torch.long)
                if faces.ndim != 2:
                    raise ValueError(f"'face' should be 2D, got {faces.shape}")
                # unify to [3, num_faces]
                if faces.size(0) == 3:
                    face = faces
                elif faces.size(1) == 3:
                    face = faces.t().contiguous()
                else:
                    raise ValueError(f"Unexpected 'face' shape {faces.shape}, expect (*,3) or (3,*)")

                data = Data(y=y, pos=pos, face=face)

            elif is_volume:
                # ---- volume / velocity: optionally concatenate pressure ----
                vel_key = _first_key(g, ['velocity', 'u', 'vel'])
                pos_key = _first_key(g, ['pos_tets', 'pos', 'points', 'vertices'])
                tet_key = _first_key(g, ['tets', 'tetra', 'cells', 'tetrahedra'])

                pos = torch.as_tensor(g[pos_key][()], dtype=torch.float32)           # [N,3]
                vel = torch.as_tensor(g[vel_key][()], dtype=torch.float32)           # [N,3]
                # unify tets to [4, num_tets]
                t_np = g[tet_key][()]
                t = torch.as_tensor(t_np, dtype=torch.long)
                if t.ndim != 2:
                    raise ValueError(f"'tets' should be 2D, got {t.shape}")
                if t.size(0) == 4:
                    tets = t
                elif t.size(1) == 4:
                    tets = t.t().contiguous()
                else:
                    raise ValueError(f"Unexpected 'tets' shape {t.shape}, expect (*,4) or (4,*)")

                # if pressure exists, concatenate to [N,4], otherwise keep [N,3]
                if 'pressure' in g or 'p' in g:
                    pres_key = _first_key(g, ['pressure', 'p'])
                    pressure = torch.as_tensor(g[pres_key][()], dtype=torch.float32)
                    if pressure.ndim == 1:
                        pressure = pressure.view(-1, 1)
                    y = torch.cat([pressure, vel], dim=1)                            # [N,4]
                else:
                    y = vel                                                           # [N,3]

                data = Data(y=y, pos=pos, tets=tets)

            else:
                raise KeyError(
                    f"Sample '{sample_id}' lacks expected task keys. Found: {sorted(keys)}; "
                    "need one of surface{wss,..} or volume{velocity,..}."
                )

            # ======== Attach boundary condition indices (optional, add if available) ========
            for k_h5, k_attr in [('inlet_idcs', 'inlet_index'),
                                ('lumen_wall_idcs', 'lumen_wall_index'),
                                ('outlets_idcs', 'outlets_index')]:
                if k_h5 in g:
                    idx = torch.as_tensor(g[k_h5][()], dtype=torch.long).view(-1)
                    # remove negative values and duplicates
                    if idx.numel() > 0:
                        idx = torch.unique(idx[idx >= 0])
                    setattr(data, k_attr, idx)

            return data



    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]), weights_only=False)

        return data
