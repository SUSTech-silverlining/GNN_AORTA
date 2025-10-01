import torch
import torch_geometric
from torch.utils.checkpoint import checkpoint
import torch_geometric.nn as gnn

class ResBlock(torch.nn.Module):

    def __init__(self, convolution, in_channels, out_channels, **kwargs):
        super(ResBlock, self).__init__()

        if 'layer_norm' in kwargs:
            if not kwargs.pop('layer_norm'):
                self.ln1 = torch.nn.Identity()
        else:
            self.ln1 = gnn.LayerNorm(out_channels, mode='node')

        if 'relu' in kwargs:
            if not kwargs.pop('relu'):
                self.act = torch.nn.Identity()
        else:
            self.act = torch.relu

        self.conv0 = convolution(in_channels, out_channels, **kwargs)
        self.ln0 = gnn.LayerNorm(out_channels, mode='node')
        self.conv1 = convolution(out_channels, out_channels, **kwargs)

        if in_channels != out_channels:
            self.lin = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.lin = torch.nn.Identity()

    @staticmethod
    def dummy_conv(conv, signal, connectivity, dummy):

        # Dummy wrapper to trick the checkpoint to preserve gradients
        return conv(signal, connectivity)

    def layer(self, x, edge_index, conv):

        # Re-calculate the convolution on each pass
        check = checkpoint(self.dummy_conv, conv, x, edge_index,
                           torch.tensor(0., requires_grad=True),
                           preserve_rng_state=False,
                           use_reentrant=False)

        return check  # without activation

    def forward(self, x, edge_index):
        y = self.layer(x, edge_index, self.conv0)
        y = self.ln0(y)
        y = torch.relu(y)
        y = self.layer(y, edge_index, self.conv1)

        # Residual connection
        out = y + self.lin(x.unsqueeze(2)).squeeze(2)
        out = self.ln1(out)
        out = self.act(out)

        return out.squeeze()


class FeaStResBlock(ResBlock):

    def __init__(self, in_channels, out_channels, **kwargs):
        convolution = gnn.FeaStConv

        super(FeaStResBlock, self).__init__(convolution, in_channels, out_channels, **kwargs)


class SAGEResBlock(ResBlock):

    def __init__(self, in_channels, out_channels, **kwargs):
        convolution = gnn.SAGEConv

        super(SAGEResBlock, self).__init__(convolution, in_channels, out_channels, **kwargs)