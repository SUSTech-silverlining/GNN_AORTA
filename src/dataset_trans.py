import os
import re
import glob
import tqdm
import torch
import numpy as np

 
import warnings
from os.path import join as osj
from shutil import rmtree  # Required for deleting folders
from torch_geometric.data import InMemoryDataset

from utils import vtp2Graph_aorta
import torch_geometric as pyg


class MSData(pyg.data.Data):

    def __cat_dim__(self, key: str, value, *args, **kwargs) -> int:
        if 'index' in key or key == 'face' or key == 'tets' or 'coo' in key:
            return -1
        else:
            return 0

    def __inc__(self, key: str, value, *args, **kwargs) -> int:
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'pool_source' in key or 'interp_target' in key or 'sampling_index' in key:
            if int(key[5]) == 0:
                return self.num_nodes
            else:
                return self[f'scale{int(key[5]) - 1}_sampling_index'].size(dim=0)
        elif 'index' in key or key == 'face' or key == 'tets':
            return self.num_nodes
        elif 'pool_target' in key or 'interp_source' in key:
            return self[f'scale{key[5]}_sampling_index'].size(dim=0)
        else:
            return 0
        

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
            # 1) 读原始数据
            data, _ = vtp2Graph_aorta(raw_path)

            # suk!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # data = torch.load(raw_path, weights_only=False)

            data.orientation = data.norm
            data.scalar_feature = data.stmdist.unsqueeze(-1)
            data.pos = data.pos

            # 2) 预处理
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Normalization of output!
            if self.label == 'pressure':
                data.y = self.output_normalizer.normalize(data.y[:, 0])
                # data.y = data.y[:, 0]
                
            elif self.label == 'wss':
                data.y = self.output_normalizer.normalize(data.y[:,1:])
                # data.y = data.y[:, 1:]

            # Normalization of input!
            data.x = torch.cat(
                [
                    data.pos,
                    data.stmdist,
                    # data.stmdist.unsqueeze(-1),
                ],
                dim=1,
            )
            data.x = self.input_normalizer.normalize(data.x)

            # ✅ 添加这两行，供 LaB-GATr 使用
            data.orientation = data.norm  # orientation ← 法向量
            data.scalar_feature = data.x[:, -1:]            # scalar_feature ← stmdist
            data.pos = data.x[:, :3]  # pos ← 位置坐标


            # ---- 5) 保存时记得把 stmdist 一并写入（如果存在）----
            save_kwargs = dict(
                y=data.y,
                pos=data.pos,
                orientation=data.orientation,
                scalar_feature=data.scalar_feature,
                idx=idx,
            )
            
            

            # ✅ 把 transform 生成的所有 scale* 字段写入（包含 sampling_index / pool_source / pool_target）
            for k in data.keys():
                if k.startswith("scale") and (
                    k.endswith("edge_index")
                    or k.endswith("cluster_map")
                    or k.endswith("sampling_index")
                    or k.endswith("pool_source")
                    or k.endswith("pool_target")
                    or k.endswith("interp_source")  # <-- 新增
                    or k.endswith("interp_target") 
                ):
                    save_kwargs[k] = data[k]

            data = MSData(**{k: v for k, v in save_kwargs.items() if v is not None})
            torch.save(data, os.path.join(self.processed_dir, f"data_{idx}.pt"))
            idx += 1




    def get(self, idx):
        data = torch.load(
            os.path.join(self.processed_dir, self.processed_file_names[idx]),
            weights_only=False  # ✅ 加在这里，解决 PyTorch 2.6+ 加载失败问题
        )
        data.to(self.device)
        return data


    def len(self):
        return len(self.raw_file_names)
