import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
from torch_geometric import transforms

from .gnn_layer import MLP, MGNConv, AttenResBlock
from .pooling import ParallelTransportPool
from .nn_utils import parameter_table
from Bench_Heat.MS_transform.scale_mask import ScaleMask

############## This part is for sigle scale baselines ####################
class MGN(torch.nn.Module):
    def __init__(
        self,
        in_node_channels:int=4,
        in_edge_channels:int=4,
        out_node_channels:int=1,
        hidFeature:int=40,
        mp_time:int=15,
    ):
        super(MGN, self).__init__()

        self.in_node_channels = in_node_channels
        self.out_node_channels = out_node_channels
        self.mp_time = mp_time

        # LocalEncoder
        self.encoderNode = MLP(
            input_dim=in_node_channels,
            latent_dim=hidFeature,
            output_dim=hidFeature,

        )
        self.encoderNodeNorm = gnn.LayerNorm(hidFeature)
        
        self.encoderEdge = MLP(
            input_dim=in_edge_channels,
            latent_dim=hidFeature,
            output_dim=hidFeature,

        )
        self.encoderEdgeNorm = gnn.LayerNorm(hidFeature)



        modules = []
        for _ in range(mp_time):
            modules.append(
                MGNConv(
                    in_node_channels=hidFeature,
                    out_node_channels=hidFeature,
                    in_edge_channels=hidFeature,
                    out_edge_channels=hidFeature,
                    latent_channels=hidFeature,
                )
            )
        self.hiddenlayers = nn.Sequential(*modules)
        self.decoder = MLP(
            input_dim=hidFeature,
            output_dim=out_node_channels,
            latent_dim=hidFeature,

        )
        print(
            "{} ({} trainable parameters)".format(
                self.__class__.__name__, self.count_parameters
            )
        )

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_table(self):
        return parameter_table(self)

    def forward(self, data:Data):
        x = data.x
        edge_index = data.edge_index
        pos = data.pos
        xij = pos[edge_index[1, :]] - pos[edge_index[0, :]]
        xij_norm = torch.norm(xij, dim=1).view(-1, 1)
        # Replace division by zero with zero
        edge_attr = torch.concat(
            [
                torch.where(xij_norm == 0, torch.tensor(0.0), xij / xij_norm),
                xij_norm,
            ],
            dim=1,
        )
        x = self.encoderNode(x)
        x = self.encoderNodeNorm(x)
        
        edge_attr = self.encoderEdge(edge_attr)
        edge_attr = self.encoderEdgeNorm(edge_attr)

        for i in range(self.mp_time):
            x = self.hiddenlayers[i](x, edge_index, edge_attr)
        x = self.decoder(x)

        return x


class BaselineArchitecture(nn.Module):
    """
    Base class for multi-scale GNN models.

    This class provides common functionality like parameter counting and
    input preparation for multi-scale graph data. The specific network
    architecture should be implemented in child classes.

    Args:
        in_channels (int): Number of input feature channels per node.
        out_channels (int): Number of output feature channels per node.
        hidden_channels (int, optional): Number of channels in hidden layers.
        num_scales (int, optional): Number of scales for pooling/unpooling.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_scales: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_scales = num_scales

        # Create a list of transform functions to generate each graph scale
        self.scale_transforms = [
            transforms.Compose([ScaleMask(i)]) for i in range(num_scales)
        ]

    @property
    def count_parameters(self) -> int:
        """Computes the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_table(self):
        """Returns a formatted table of model parameters for inspection."""
        return parameter_table(self)

    def prepare_input(self, data: Data) -> list:
        """
        Applies scale transforms to the input graph to get edge indices for each scale.

        Args:
            data (Data): Input graph data.

        Returns:
            List[torch.Tensor]: A list of edge_index tensors, one for each scale.
        """
        scale_data = [s(data) for s in self.scale_transforms]
        return [d.edge_index for d in scale_data]


class AttGCN(BaselineArchitecture):
    """
    A flexible multi-scale Graph Convolutional Network with a U-Net architecture.

    This model dynamically constructs a series of encoding, pooling, unpooling, and
    decoding blocks based on the specified number of scales and message passing steps.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        hidden_channels (int, optional): Number of hidden channels. Defaults to 64.
        num_scales (int, optional): The number of scales (depth of the U-Net). Defaults to 3.
        down_mp_times (int, optional): Number of message passing blocks in the encoder/downstream path. Defaults to 2.
        up_mp_times (int, optional): Number of message passing blocks in the upstream/decoder path. Defaults to 4.
        add_self_loops (bool, optional): Whether to add self-loops in the attention mechanism. Defaults to False.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 94,
        num_scales: int = 3,
        down_mp_times: int = 2,
        up_mp_times: int = 4,
        add_self_loops: bool = False,
    ):
        super().__init__(in_channels, out_channels, hidden_channels, num_scales)
        self.down_mp_times = down_mp_times
        self.up_mp_times = up_mp_times
        kwargs = dict(add_self_loops=add_self_loops, heads=4)

        # --- Dynamic Layer Creation ---
        self.encoder_convs = nn.ModuleList()
        self.downstream_convs = nn.ModuleList()
        self.upstream_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()

        # 1. Encoder (Initial convolutions at scale 0)
        self.encoder_convs.append(AttenResBlock(in_channels, hidden_channels, **kwargs))
        for _ in range(down_mp_times):
            self.encoder_convs.append(AttenResBlock(hidden_channels, hidden_channels, **kwargs))

        # 2. Downstream Path (Pooling and convolutions for scales 1 to num_scales-1)
        for i in range(num_scales - 1):
            self.pools.append(ParallelTransportPool(i + 1, unpool=False, latent_dim=hidden_channels))
            
            convs = nn.ModuleList()
            for _ in range(down_mp_times):
                convs.append(AttenResBlock(hidden_channels, hidden_channels, **kwargs))
            self.downstream_convs.append(convs)

        # 3. Upstream Path (Unpooling and convolutions from the bottleneck back to scale 0)
        for i in range(num_scales - 1):
            scale_level = (num_scales - 1) - i
            self.unpools.append(ParallelTransportPool(scale_level, unpool=True, latent_dim=hidden_channels))

            convs = nn.ModuleList()
            # First conv block after unpooling takes concatenated features from skip connection
            convs.append(AttenResBlock(hidden_channels * 2, hidden_channels, **kwargs))
            
            # Add remaining convolution blocks for this scale
            for j in range(up_mp_times - 1):
                # Special case for the final block of the entire network
                is_final_block = (i == num_scales - 2) and (j == up_mp_times - 2)
                if is_final_block:
                    convs.append(AttenResBlock(hidden_channels, out_channels, layer_norm=False, ReLU=False, **kwargs))
                else:
                    convs.append(AttenResBlock(hidden_channels, hidden_channels, **kwargs))
            self.upstream_convs.append(convs)
        print(f"{self.__class__.__name__} ({self.count_parameters} trainable parameters) initialized.")

    def forward(self, data: Data) -> torch.Tensor:
        """
        Defines the forward pass of the U-Net GNN.

        Args:
            data (Data): The input graph data object.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        x = data.x
        scale_attr = self.prepare_input(data)
        skip_connections = []

        # === Encoder and Downstream Path ===
        # Apply initial encoder blocks at scale 0
        for conv in self.encoder_convs:
            x = conv(x, scale_attr[0])
        skip_connections.append(x)

        # Apply pooling and convolution blocks for each subsequent scale
        for i in range(self.num_scales - 1):
            x = self.pools[i](data, x)
            for conv in self.downstream_convs[i]:
                x = conv(x, scale_attr[i + 1])
            # Store feature map for skip connection, except for the last one (bottleneck)
            if i < self.num_scales - 2:
                skip_connections.append(x)

        # === Upstream and Decoder Path ===
        # Apply unpooling, skip connections, and convolutions to go back up
        for i in range(self.num_scales - 1):
            x = self.unpools[i](data, x)
            skip_x = skip_connections.pop()
            x = torch.cat((x, skip_x), dim=1) # Apply skip connection
            for conv in self.upstream_convs[i]:
                x = conv(x, scale_attr[(self.num_scales - 2) - i])

        return x.squeeze()