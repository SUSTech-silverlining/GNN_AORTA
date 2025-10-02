import os
import glob
import torch
import pyvista as pv
import matplotlib.pyplot as plt
from os.path import join as osj
from tqdm import tqdm
from collections.abc import Iterable
from math import log, cos, pi, floor
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.regression import RelativeSquaredError


def train(model, device, train_loader, optimizer, criterion, scheduler=None):
    model.train()
    train_loss = torch.zeros(len(train_loader), device=device)
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        output = model(batch)
        # label = batch.y
        label  = batch.y[:, 0].view(-1, 1)      # pressure
        loss = criterion(output, label)  # masked optimisation
        train_loss[i] = loss.clone()
        loss.backward()
        optimizer.step()
        if scheduler != None:
            scheduler.step()
    return torch.mean(train_loss)


def valid(model, device, val_loader, criterion):
    model.eval()
    valid_loss = torch.zeros(len(val_loader), device=device)
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = batch.to(device)
            # label = batch.y
            label  = batch.y[:, 0].view(-1, 1) 
            output = model(batch)
            loss = criterion(output, label)
            valid_loss[i] = loss.clone()

        val_batch_error = torch.mean(valid_loss)
    return val_batch_error


def test_output(
    model,
    device,
    output_loader,
    output_normalizer,
    name="",
    need_save=True,
    dataset="train",
):
    model.eval()
    RSE_loss = torch.zeros(len(output_loader))
    for i, batch in enumerate(tqdm(output_loader)):
        batch = batch.to(device)
        with torch.no_grad():
            # label = batch.y
            label  = batch.y[:, 0].view(-1, 1) 
            output = model(batch)

            label_denorm = output_normalizer.denormalize(label)  # shape is <N,1>
            output_denorm = output_normalizer.denormalize(output)  # shape is <N,1>
            # Calculate the mean of the labels tensor
            relative_squared_error = RelativeSquaredError(label.shape[1])
            RSE = relative_squared_error(
                output_denorm.detach().cpu(), label_denorm.detach().cpu()
            )
            RSE_loss[i] = RSE.clone()

    print("max RMSE", torch.max(RSE_loss))
    print("RMSE mean", torch.mean(RSE_loss))
    print("max RMAE", torch.sqrt(torch.max(RSE_loss)))
    print("RMAE mean", torch.sqrt(torch.mean(RSE_loss)))

    plt.hist(RSE_loss.numpy(), bins=15, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of RMSE on {dataset}")
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.grid(True)
    if need_save:
        plt.savefig("results/figs/" + name + "_RMSE.png", bbox_inches="tight")


def generate_output(
    model,
    device,
    output_loader,
    raw_mesh_path,
    tag,
    save_mesh_path,
    mesh_name="mesh",
    mesh_dim: int = 2,
    output_normalizer=None,
    need_duplicate=False,
    need_Flatten=True
):
    for _, batch in enumerate(tqdm(output_loader)):
        batch = batch.to(device)
        with torch.no_grad():
            # label = batch.y
            label  = batch.y[:, 0].view(-1, 1) 
            output = model(batch)
            label_denorm = output_normalizer.denormalize(label)  # shape is <N,1>
            output_denorm = output_normalizer.denormalize(output)  # shape is <N,1>
            idx = (batch.idx)[0]

            # Construct the search pattern for .vtp files
            search_pattern = os.path.join(raw_mesh_path, "*.vtp")

            # Find all .vtp files in the specified directory
            vtp_files = glob.glob(search_pattern)

            # Check if the list of .vtp files is empty
            if not vtp_files:
                print(
                    f"No .vtp files found in the specified directory:{raw_mesh_path}, will find them in case dirs"
                )
                mesh = pv.read(
                    osj(
                        raw_mesh_path + f"/case-{idx}",
                        "converted_point_data{:d}.vtp".format(idx),
                    )
                )

            else:
                mesh = pv.read(raw_mesh_path + "/"+mesh_name+"{:d}.vtp".format(idx)

                )
                print(f"Found .vtp file(s).")
            output_denorm = output_denorm.detach().cpu()
            label_denorm = label_denorm.detach().cpu()

            if mesh_dim == 2:
                if need_duplicate:
                    output_denorm = torch.cat((output_denorm, output_denorm), dim=0)
                    label_denorm = torch.cat((label_denorm, label_denorm), dim=0)
                if need_Flatten:
                    points = mesh.points
                    points[:, 2] = 0
                    mesh.points = points
                
            
            mesh.point_data[tag + "_prediction"] = output_denorm
            mesh.point_data[tag + "_label"] = label_denorm
            mesh.point_data[tag + "_error"] = (
                mesh.point_data[tag + "_prediction"] - mesh.point_data[tag + "_label"]
            )
            if not os.path.exists(save_mesh_path):
                # Create the directory
                os.makedirs(save_mesh_path)
                print(f"Directory created: {save_mesh_path}")
            else:
                print(f"Directory already exists: {save_mesh_path}")
            mesh.save(osj(save_mesh_path, mesh_name + "_{:d}.vtp".format(idx)))


def graphid2mesh(return_samples, mesh_path, tag, save_path):
    """
    input: return_samples = {[i,output_denorm,label_denorm]}
    output: mesh vtp file, mesh obj.
    """

    for i, ele in enumerate(return_samples):
        mesh = pv.read(osj(mesh_path, "data{:d}.vtp".format(ele[0])))
        mesh.point_data[tag + "_prediction"] = ele[1].detach()
        mesh.point_data[tag + "_label"] = ele[2].detach()
        mesh.point_data[tag + "_error"] = (ele[1] - ele[2]).detach()
        mesh.save(osj(save_path, "mesh_{:d}.vtp".format(ele[0])))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def cal_epoches(
    ii,
    init_decay_epochs=500,
    restart_interval=300,
    warmup_epochs=200,
    restart_interval_multiplier=1.5,
):
    epoch = init_decay_epochs + warmup_epochs
    for i in range(ii):
        epoch += int(restart_interval * restart_interval_multiplier)
    return epoch


def epoch2cosepcoh(my_epoch, **kargs):
    ii = 0
    epoch = 0
    while epoch <= my_epoch:
        ii += 1
        epoch = cal_epoches(ii, **kargs)
    return cal_epoches(ii, **kargs)


class CyclicCosineDecayLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        init_decay_epochs,
        min_decay_lr,
        restart_interval=None,
        restart_interval_multiplier=None,
        restart_lr=None,
        warmup_epochs=None,
        warmup_start_lr=None,
        last_epoch=-1,
        verbose=False,
    ):
        """
        Initialize new CyclicCosineDecayLR object.
        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_decay_epochs: (int) - Number of initial decay epochs.
        :param min_decay_lr: (float or iterable of floats) - Learning rate at the end of decay.
        :param restart_interval: (int) - Restart interval for fixed cycles.
            Set to None to disable cycles. Default: None.
        :param restart_interval_multiplier: (float) - Multiplication coefficient for geometrically increasing cycles.
            Default: None.
        :param restart_lr: (float or iterable of floats) - Learning rate when cycle restarts.
            If None, optimizer's learning rate will be used. Default: None.
        :param warmup_epochs: (int) - Number of warmup epochs. Set to None to disable warmup. Default: None.
        :param warmup_start_lr: (float or iterable of floats) - Learning rate at the beginning of warmup.
            Must be set if warmup_epochs is not None. Default: None.
        :param last_epoch: (int) - The index of the last epoch. This parameter is used when resuming a training job. Default: -1.
        :param verbose: (bool) - If True, prints a message to stdout for each update. Default: False.
        """

        if not isinstance(init_decay_epochs, int) or init_decay_epochs < 1:
            raise ValueError(
                "init_decay_epochs must be positive integer, got {} instead".format(
                    init_decay_epochs
                )
            )

        if isinstance(min_decay_lr, Iterable) and len(min_decay_lr) != len(
            optimizer.param_groups
        ):
            raise ValueError(
                "Expected len(min_decay_lr) to be equal to len(optimizer.param_groups), "
                "got {} and {} instead".format(
                    len(min_decay_lr), len(optimizer.param_groups)
                )
            )

        if restart_interval is not None and (
            not isinstance(restart_interval, int) or restart_interval < 1
        ):
            raise ValueError(
                "restart_interval must be positive integer, got {} instead".format(
                    restart_interval
                )
            )

        if restart_interval_multiplier is not None and (
            not isinstance(restart_interval_multiplier, float)
            or restart_interval_multiplier <= 0
        ):
            raise ValueError(
                "restart_interval_multiplier must be positive float, got {} instead".format(
                    restart_interval_multiplier
                )
            )

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(
            optimizer.param_groups
        ):
            raise ValueError(
                "Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                "got {} and {} instead".format(
                    len(restart_lr), len(optimizer.param_groups)
                )
            )

        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
                raise ValueError(
                    "Expected warmup_epochs to be positive integer, got {} instead".format(
                        type(warmup_epochs)
                    )
                )

            if warmup_start_lr is None:
                raise ValueError(
                    "warmup_start_lr must be set when warmup_epochs is not None"
                )

            if not (
                isinstance(warmup_start_lr, float)
                or isinstance(warmup_start_lr, Iterable)
            ):
                raise ValueError(
                    "warmup_start_lr must be either float or iterable of floats, got {} instead".format(
                        warmup_start_lr
                    )
                )

            if isinstance(warmup_start_lr, Iterable) and len(warmup_start_lr) != len(
                optimizer.param_groups
            ):
                raise ValueError(
                    "Expected len(warmup_start_lr) to be equal to len(optimizer.param_groups), "
                    "got {} and {} instead".format(
                        len(warmup_start_lr), len(optimizer.param_groups)
                    )
                )

        group_num = len(optimizer.param_groups)
        self._warmup_start_lr = (
            [warmup_start_lr] * group_num
            if isinstance(warmup_start_lr, float)
            else warmup_start_lr
        )
        self._warmup_epochs = 0 if warmup_epochs is None else warmup_epochs
        self._init_decay_epochs = init_decay_epochs
        self._min_decay_lr = (
            [min_decay_lr] * group_num
            if isinstance(min_decay_lr, float)
            else min_decay_lr
        )
        self._restart_lr = (
            [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        )
        self._restart_interval = restart_interval
        self._restart_interval_multiplier = restart_interval_multiplier
        super(CyclicCosineDecayLR, self).__init__(
            optimizer, last_epoch, verbose=verbose
        )

    def get_lr(self):

        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return self._calc(
                self.last_epoch,
                self._warmup_epochs,
                self._warmup_start_lr,
                self.base_lrs,
            )

        elif self.last_epoch < self._init_decay_epochs + self._warmup_epochs:
            return self._calc(
                self.last_epoch - self._warmup_epochs,
                self._init_decay_epochs,
                self.base_lrs,
                self._min_decay_lr,
            )
        else:
            if self._restart_interval is not None:
                if self._restart_interval_multiplier is None:
                    cycle_epoch = (
                        self.last_epoch - self._init_decay_epochs - self._warmup_epochs
                    ) % self._restart_interval
                    lrs = (
                        self.base_lrs if self._restart_lr is None else self._restart_lr
                    )
                    return self._calc(
                        cycle_epoch, self._restart_interval, lrs, self._min_decay_lr
                    )
                else:
                    n = self._get_n(
                        self.last_epoch - self._warmup_epochs - self._init_decay_epochs
                    )
                    sn_prev = self._partial_sum(n)
                    cycle_epoch = (
                        self.last_epoch
                        - sn_prev
                        - self._warmup_epochs
                        - self._init_decay_epochs
                    )
                    interval = (
                        self._restart_interval * self._restart_interval_multiplier**n
                    )
                    lrs = (
                        self.base_lrs if self._restart_lr is None else self._restart_lr
                    )
                    return self._calc(cycle_epoch, interval, lrs, self._min_decay_lr)
            else:
                return self._min_decay_lr

    def _calc(self, t, T, lrs, min_lrs):
        return [
            min_lr + (lr - min_lr) * ((1 + cos(pi * t / T)) / 2)
            for lr, min_lr in zip(lrs, min_lrs)
        ]

    def _get_n(self, epoch):
        _t = (
            1 - (1 - self._restart_interval_multiplier) * epoch / self._restart_interval
        )
        return floor(log(_t, self._restart_interval_multiplier))

    def _partial_sum(self, n):
        return (
            self._restart_interval
            * (1 - self._restart_interval_multiplier**n)
            / (1 - self._restart_interval_multiplier)
        )
