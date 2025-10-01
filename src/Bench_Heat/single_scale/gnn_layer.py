import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch.utils.checkpoint import checkpoint



class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        output_dim: int,
        hidden_layers: int=2,

    ):
        super(MLP, self).__init__()
        modules = []
        for l in range(hidden_layers):
            if l == 0:
                modules.append(nn.Linear(input_dim, latent_dim))
            else:
                modules.append(nn.Linear(latent_dim, latent_dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(latent_dim, output_dim))

        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class MGNConv(gnn.MessagePassing):
    r"""
    A custom Message Passing Neural Network layer for MGN.
    """

    # MGN layers
    # the in_edge_channels should be equal to out_edge_channels!!!
    def __init__(
        self,
        in_node_channels: int,
        out_node_channels: int,
        in_edge_channels: int,
        out_edge_channels: int,
        latent_channels: int,
    ):
        super(MGNConv, self).__init__(aggr="mean")
        self.in_node_channels = in_node_channels
        self.out_node_channels = out_node_channels
        self.in_edge_channels = in_edge_channels
        self.out_edge_channels = out_edge_channels

        self.nnEdge = MLP(
            input_dim=2 * self.in_node_channels
            + self.in_edge_channels,  # in_dim=[xi,xj,edge_dim]
            output_dim=self.out_edge_channels,  # out_dim=edge_dim
            latent_dim=latent_channels,
        )
        self.normEdge = gnn.LayerNorm(self.out_edge_channels)
        
        self.nnNode = MLP(
            input_dim=self.in_node_channels
            + self.out_edge_channels,  # in_dim=[xi,sum_of_edge_feat]
            output_dim=self.out_node_channels,
            latent_dim=latent_channels,
        )
        self.normNode = gnn.LayerNorm(self.out_node_channels)
        # Handling the residual connection
        if in_node_channels != out_node_channels:
            self.lin = torch.nn.Conv1d(in_node_channels, out_node_channels, kernel_size=1)
        else:
            self.lin = torch.nn.Identity()

    def forward(self, x0, edge_index, edge_attr):
        out = self.propagate(
            edge_index, x=x0, edge_attr=edge_attr
        )  # do one time message passing
        out = self.normNode(self.nnNode((torch.cat([x0, out], dim=1))))
        x0 = self.lin(x0.unsqueeze(2)).squeeze(2)
        x = x0 + out
        return x

    def message(self, x_i, x_j, edge_attr):
        tmp_edge = self.nnEdge((torch.cat([x_i, x_j, edge_attr], dim=1)))
        tmp_edge = self.normEdge(tmp_edge)
        edge_attr = tmp_edge + edge_attr
        return edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(node_dim={self.in_node_channels}, edge_dim={self.in_edge_channels})"


class ResBlock(nn.Module):

    def __init__(self, convolution, in_node_channels, out_node_channels, **kwargs):
        super(ResBlock, self).__init__()

        if "layer_norm" in kwargs:
            if not kwargs.pop("layer_norm"):
                self.ln1 = torch.nn.Identity()
        else:
            self.ln1 = gnn.LayerNorm(out_node_channels)

        if "ReLU" in kwargs:
            if not kwargs.pop("ReLU"):
                self.act = nn.Identity()
        else:
            self.act = nn.ReLU()

        self.conv0 = convolution(in_node_channels, out_node_channels, **kwargs)
        self.ln0 = gnn.LayerNorm(out_node_channels)
        self.conv1 = convolution(out_node_channels, out_node_channels, **kwargs)

        if in_node_channels != out_node_channels:
            self.lin = nn.Conv1d(in_node_channels, out_node_channels, kernel_size=1)
        else:
            self.lin = nn.Identity()

    @staticmethod
    def dummy_conv(conv, signal, connectivity, dummy):

        # Dummy wrapper to trick the checkpoint to preserve gradients
        return conv(signal, connectivity)

    def layer(self, x, edge_index, conv):

        # Re-calculate the convolution on each pass
        check = checkpoint(
            self.dummy_conv,
            conv,
            x,
            edge_index,
            torch.tensor(0.0, requires_grad=True),
            preserve_rng_state=False,
            use_reentrant=False,
        )

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


class AttenResBlock(ResBlock):
    def __init__(self, in_node_channels, out_node_channels, **kwargs):
        convolution = gnn.FeaStConv
        super(AttenResBlock, self).__init__(
            convolution, in_node_channels, out_node_channels, **kwargs
        )
