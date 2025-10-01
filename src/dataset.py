import os
import re
import glob
import tqdm
import torch
import numpy as np


import warnings
from os.path import join as osj
from shutil import rmtree  # Required for deleting folders
from torch_geometric.data import InMemoryDataset, Data

from utils import vtp2Graph_poisson, vtp2Graph_aorta
from Bench_Heat.MS_data import MultiscaleData 

class Aorta_3d_Dataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        device:str,
        label:str,
        input_normalizer=None,
        output_normalizer=None,
        transform=None,
        pre_transform=None,
    ):
        # Initialize parameters
        self.root = root
        self.device = device
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.label = label

        # Determine filenames and sort them
        filename = [
            os.path.basename(ele) for ele in glob.glob(os.path.join(root, "raw/*"))
        ]
        filename.sort(key=lambda f: int(re.sub("\D", "", f)))
        self.filename = filename
        # self.Encoder2 = PosEncodingNeRF(in_features=1,fn_samples=10000)
        # # Perform pre-transform consistency check
        # if os.path.exists(self.processed_dir):
        #     existing_pre_transform = self._load_existing_pre_transform()
        #     if (
        #         existing_pre_transform is not None
        #         and existing_pre_transform != pre_transform
        #     ):
        #         warnings.warn(
        #             "The `pre_transform` argument differs from the one used in the pre-processed version of this dataset. "
        #             "The existing processed folder will be deleted to ensure consistency."
        #         )
        #         # Delete the processed folder to allow re-processing with new pre-transform
        #         rmtree(self.processed_dir)

        # Call the superclass constructor
        super().__init__(root, transform, pre_transform)

    def _load_existing_pre_transform(self):
        """Load and return the pre-transform used in the existing processed data."""
        pre_transform_path = os.path.join(self.processed_dir, "pre_transform.pt")
        if os.path.exists(pre_transform_path):
            existing_pre_transform = torch.load(pre_transform_path)
            return existing_pre_transform
        return None

    @property
    def raw_file_names(self):
        return self.filename


    @property
    def processed_file_names(self):
        return [f"data_{i}.pt" for i in range(self.len())]


    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_path in tqdm.tqdm(self.raw_paths):
            # Convert VTP to graph
            data, _ = vtp2Graph_aorta(raw_path)

            # # suk!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # data = torch.load(raw_path, weights_only=False)

            # "y": "pressure, wssx, wssy, wssz",
            if self.label == 'pressure':
                data.y = self.output_normalizer.normalize(data.y[:,0])
                # data.y = data.y[:,0]
            elif self.label == 'wss':
                data.y = self.output_normalizer.normalize(data.y[:,1:])
                # data.y = data.y[:,1:]
            
            print(f"Shape of data.pos: {data.pos.shape}")
            print(f"Shape of data.stmdist: {data.stmdist.shape}")

            data.x = torch.cat(
                [
                    data.pos,
                    # data.stmdist.unsqueeze(-1),
                    data.stmdist,
                ],
                dim=1,
            )
            data.x = self.input_normalizer.normalize(data.x)

            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            extra_attrs = {}
            for k in data.keys():
                if k.startswith('scale'):
                    extra_attrs[k] = data[k]

            data = MultiscaleData(
                x=data.x,
                edge_index=data.edge_index,
                y=data.y,
                pos=data.pos,
                norm=data.norm,
                face=data.face,
                **extra_attrs
            )

            
            torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1

    def get(self, idx):
        data = torch.load(
            os.path.join(self.processed_dir, self.processed_file_names[idx]),
            weights_only=False 
        )
        
        data.to(self.device)
        return data

    def len(self):
        return len(self.raw_file_names)


class PoissonDataset(InMemoryDataset):
    def __init__(
        self,
        root:str,
        device:str,
        input_normalizer=None,
        output_normalizer=None,
        transform=None,
        pre_transform=None,
    ):

        filename = [
            os.path.basename(ele) for ele in glob.glob(os.path.join(root, "raw/*"))
        ]
        filename.sort(key=lambda f: int(re.sub("\D", "", f)))
        self.filename = filename
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.device = device
        super(PoissonDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):

        return self.filename

    @property
    def processed_file_names(self):

        return ["data_{:d}.pt".format(i) for i in range(self.len())]

    def download(self):
        return

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # converting vtp to graph
            data = vtp2Graph_poisson(raw_path)
            data.x = self.input_normalizer.normalize(data.source)
            data.y = self.output_normalizer.normalize(data.temperature)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

                data = Data(
                    x=data.x,
                    edge_index=data.edge_index,
                    y=data.y,
                    pos=data.pos,
                    edge_mask=data.edge_mask,
                    node_mask=data.node_mask,
                )
            else:
                data = Data(
                    x=data.x,
                    edge_index=data.edge_index,
                    y=data.y,
                    pos=data.pos,
                )

            data.idx = idx
            torch.save(data, osj(self.processed_dir, "data_{:d}.pt".format(idx)))
            idx += 1

    def get(self, idx):
        data = torch.load(osj(self.processed_dir, self.processed_file_names[idx]))
        data.to(self.device)
        return data

    def to(self, device):
        self._data.to(device)
        return self

    def len(self):
        return len(self.raw_file_names)