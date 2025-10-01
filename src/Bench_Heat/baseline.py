import torch
from utils import parameter_table
from Bench_Heat.baseline_res_block import SAGEResBlock, FeaStResBlock
from Bench_Heat.cluster_pooling import ClusterPooling


# Base class for different convolutions
class BaselineArchitecture(torch.nn.Module):
    def __init__(self):
        super(BaselineArchitecture, self).__init__()

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_table(self):
        return parameter_table(self)

    def forward(self, data):
        # (N, 3, 3, 3) -> [(N, 6), (N, 6), (N, 9)] (two matrices are symmetric)
        # tmp = [data.feat[:, 0][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        #        data.feat[:, 1][:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]],
        #        data.feat[:, 2].reshape(-1, 9)]
        # data.x = torch.hstack((torch.hstack(tmp), data.clone().geo.unsqueeze(1)))

        # Encoder
        data.x = self.conv01(data.x, data.scale0_edge_index)
        data.x = self.conv02(data.x, data.scale0_edge_index)

        # Downstream
        copy0 = data.x.clone()
        data = self.pool1(data)
        data.x = self.conv11(data.x, data.edge_index)
        data.x = self.conv12(data.x, data.edge_index)

        copy1 = data.x.clone()
        data = self.pool2(data)
        data.x = self.conv21(data.x, data.edge_index)
        data.x = self.conv22(data.x, data.edge_index)

        # Upstream
        data = self.pool2.unpool(data)
        data.x = torch.cat((data.x, copy1), dim=1)  # "copy/cat"
        data.x = self.conv13(data.x, data.edge_index)
        data.x = self.conv14(data.x, data.edge_index)
        data.x = self.conv15(data.x, data.edge_index)
        data.x = self.conv16(data.x, data.edge_index)

        # Decoder
        data = self.pool1.unpool(data)
        data.x = torch.cat((data.x, copy0), dim=1)  # "copy/cat"
        data.x = self.conv03(data.x, data.edge_index)
        data.x = self.conv04(data.x, data.edge_index)
        data.x = self.conv05(data.x, data.edge_index)
        data.x = self.conv06(data.x, data.edge_index)

        # return data.x.squeeze()
        return data.x.view(-1, 1)  # 保留二维形状



# Attention-scaled graph convolutional (residual) network for comparison with GEM-GCN
class AttGCN(BaselineArchitecture):
    def __init__(self):
        super(AttGCN, self).__init__()

        channels = 140
        kwargs = dict(
            heads=4,
            add_self_loops=False
        )

        # Encoder
        self.conv01 = FeaStResBlock(4, channels, **kwargs)
        self.conv02 = FeaStResBlock(channels, channels, **kwargs)

        # Downstream
        self.pool1 = ClusterPooling(1)
        self.conv11 = FeaStResBlock(channels, channels, **kwargs)
        self.conv12 = FeaStResBlock(channels, channels, **kwargs)

        self.pool2 = ClusterPooling(2)
        self.conv21 = FeaStResBlock(channels, channels, **kwargs)
        self.conv22 = FeaStResBlock(channels, channels, **kwargs)

        # Up-stream
        self.conv13 = FeaStResBlock(channels + channels, channels, **kwargs)
        self.conv14 = FeaStResBlock(channels, channels, **kwargs)
        self.conv15 = FeaStResBlock(channels, channels, **kwargs)
        self.conv16 = FeaStResBlock(channels, channels, **kwargs)

        # Decoder
        self.conv03 = FeaStResBlock(channels + channels, channels, **kwargs)
        self.conv04 = FeaStResBlock(channels, channels, **kwargs)
        self.conv05 = FeaStResBlock(channels, channels, **kwargs)
        self.conv06 = FeaStResBlock(channels, 1, layer_norm=False, relu=False, **kwargs) # output channels = 1

        print("{} ({} trainable parameters)".format(self.__class__.__name__, self.count_parameters))


# Isotropic graph convolutional (residual) network for comparison with GEM-GCN
class IsoGCN(BaselineArchitecture):
    def __init__(self):
        super(IsoGCN, self).__init__()

        channels = 180
        kwargs = dict(
            root_weight=False
        )

        # Encoder
        self.conv01 = SAGEResBlock(4, channels, **kwargs)
        self.conv02 = SAGEResBlock(channels, channels, **kwargs)

        # Downstream
        self.pool1 = ClusterPooling(1)
        self.conv11 = SAGEResBlock(channels, channels, **kwargs)
        self.conv12 = SAGEResBlock(channels, channels, **kwargs)

        self.pool2 = ClusterPooling(2)
        self.conv21 = SAGEResBlock(channels, channels, **kwargs)
        self.conv22 = SAGEResBlock(channels, channels, **kwargs)

        # Up-stream
        self.conv13 = SAGEResBlock(channels + channels, channels, **kwargs)
        self.conv14 = SAGEResBlock(channels, channels, **kwargs)
        self.conv15 = SAGEResBlock(channels, channels, **kwargs)
        self.conv16 = SAGEResBlock(channels, channels, **kwargs)

        # Decoder
        self.conv03 = SAGEResBlock(channels + channels, channels, **kwargs)
        self.conv04 = SAGEResBlock(channels, channels, **kwargs)
        self.conv05 = SAGEResBlock(channels, channels, **kwargs)
        self.conv06 = SAGEResBlock(channels, 3, batch_norm=False, relu=False, **kwargs)

        print("{} ({} trainable parameters)".format(self.__class__.__name__, self.count_parameters))