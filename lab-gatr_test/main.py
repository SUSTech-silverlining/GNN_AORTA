import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# 获取当前文件的绝对路径
current_file_path = os.getcwd()

# 跳出 g_unet 目录，进入 Xinyi_GNN_aorta/src
project_root = os.path.abspath(os.path.join(current_file_path, ".."))
src_path = os.path.join(project_root, "src")

print(f"src path: {src_path}")
sys.path.insert(0, src_path)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import aorta_3D_info, Normalizer_ts,vtp2Graph_aorta

# In this code block we will derive the data information for the aorta_toy dataset.
raw_data_path = '/ehome/xinyi/Xinyi_GNN_aorta/data/toy_trans/raw'
outputpath = '/ehome/xinyi/Xinyi_GNN_aorta/data/toy_trans'
data_label = ['y', 'pos', 'stmdist']
data_information = aorta_3D_info(path=raw_data_path, labels=data_label, output_path=outputpath)

import numpy as np

# Extracting output and input ranges from the data information for data normalization
output_range = data_information.info.values[1].reshape(1, -1) # pressure!!!!!!!!!!!!!
input_range = data_information.info.values[4:] 


output_range = output_range.astype(np.float32)
input_range = input_range.astype(np.float32)

input_normalizer = Normalizer_ts(method='ms',params=input_range[:,2:].T)
output_normalizer = Normalizer_ts(method="ms", params=output_range[:,2:].T)

from dataset_copy import Aorta_3d_Dataset

# from MS_transform.Multiscale_transform import RadiusGraph
# from torch_geometric.transforms.compose import Compose

# from shape_descriptors import MatrixFeatures
# from heat_sampling import HeatSamplingCluster

# ratios = [1.0, 0.3, 0.15]
# radii  = [1.0, 10.0, 30.0]
# schemes_rg = {'subsample':'fps', 'connect':'radius','scalemap':'knn'}
# rg_transform = RadiusGraph(schemes_rg,ratios,radii)
# # transforms = Compose([
# #     rg_transform,
# #     MatrixFeatures(r=0.05),
# #     HeatSamplingCluster(ratios, radii, loop=True, max_neighbours=512)
# # ])

from lab_gatr.transforms import PointCloudPoolingScales

import torch_geometric.transforms as T

# 1. 定义采样比例和插值简单形
sampling_ratios = (0.1,)
interpolation_simplex = 'triangle'

# 2. 正确地创建 PointCloudPoolingScales 实例
#    传入 rel_sampling_ratios 和 interp_simplex 参数
transforms = T.Compose([
    PointCloudPoolingScales(rel_sampling_ratios=sampling_ratios, interp_simplex=interpolation_simplex),
    # 其他预处理...
])

root = '/ehome/xinyi/Xinyi_GNN_aorta/data/toy_trans'

dataset = Aorta_3d_Dataset(root,'cpu', label='pressure', pre_transform=transforms,input_normalizer=input_normalizer,output_normalizer=output_normalizer)

from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

total = len(dataset)
train_size = int(0.9 * total)
valid_size = int(0.1 * total)
test_size  = total - train_size

train_dataset, test_dataset= random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
_, valid_dataset = random_split(train_dataset, [train_size-valid_size, valid_size], generator=torch.Generator().manual_seed(1))

train_loader = DataLoader(train_dataset, batch_size = 4, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size = 1)
test_loader = DataLoader(test_dataset, batch_size = 1)

from gatr.interface import embed_oriented_plane, extract_translation

class GeometricAlgebraInterface:
    num_input_channels = num_output_channels = 1
    num_input_scalars = num_output_scalars = 1

    @staticmethod
    @torch.no_grad()
    def embed(data):

        multivectors = embed_oriented_plane(normal=data.orientation, position=data.pos).view(-1, 1, 16)
        scalars = data.scalar_feature.view(-1, 1)

        return multivectors, scalars

    @staticmethod
    def dislodge(multivectors, scalars):
        return extract_translation(multivectors).squeeze()

from lab_gatr.models.lab_gatr import LaBGATr
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

geo_intf = GeometricAlgebraInterface  # 你定义的那个 class

model = LaBGATr(
    geometric_algebra_interface=geo_intf,
    d_model=64,
    num_blocks=4,
    num_attn_heads=4,
    num_latent_channels=48,   # 必须偶数
    use_class_token=False,
    dropout_probability=0.1,
).to(device)



lr = 0.01

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.001)
schedule = ExponentialLR(optimizer,.999)

optim_para = {
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':1,
            }

from tqdm import tqdm
from train import train, valid

import torch
import matplotlib.pyplot as plt
import os

# ✅ checkpoint 路径
ckpt_dir = 'checkpoints1'
os.makedirs(ckpt_dir, exist_ok=True)

# ✅ 图像保存路径
fig_dir = 'figures1'
os.makedirs(fig_dir, exist_ok=True)

# ✅ 初始化日志
train_epoch_error = []
val_epoch_error = []
test_epoch_error = []
check_idx = []

epochs = 500

# ✅ 开始训练
for epoch in tqdm(range(epochs)):
    train_loss = train(model, device, train_loader, optimizer, criterion, scheduler=schedule)
    train_epoch_error.append(train_loss)

    if (epoch  % 100) == 0:
        val_loss = valid(model, device, valid_loader, criterion)
        test_loss = valid(model, device, test_loader, criterion)

        val_epoch_error.append(val_loss)
        test_epoch_error.append(test_loss)
        check_idx.append(epoch)

        # ✅ 保存 checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss
        }, os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pt'))

        # ✅ 画图 1：训练损失
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(
            range(len(train_epoch_error)),
            torch.stack(train_epoch_error).detach().cpu().numpy(),
            color='C0', label='Train Loss'
        )
        ax1.set_yscale('log')
        ax1.set_title('Train Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(os.path.join(fig_dir, f"train_loss_epoch_{epoch}.png"))
        plt.close(fig1)

        # ✅ 画图 2：Val/Test 损失（每 100 epoch）
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(
            check_idx,
            torch.stack(val_epoch_error).detach().cpu().numpy(),
            color='C1', label='Validation Loss'
        )
        ax2.plot(
            check_idx,
            torch.stack(test_epoch_error).detach().cpu().numpy(),
            color='C2', label='Test Loss'
        )
        ax2.set_yscale('log')
        ax2.set_title('Validation & Test Loss (every 100 epochs)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(fig_dir, f"val_test_loss_epoch_{epoch}.png"))
        plt.close(fig2)
