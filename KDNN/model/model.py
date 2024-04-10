import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import Tuple, Union

from torch.nn import ModuleList, Sequential, ReLU, Linear

from torch_geometric.nn import GATConv, MetaLayer, GCNConv
from torch_geometric.nn import BatchNorm, global_mean_pool


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class EdgeConvLayer(nn.Module):

    def __init__(self,
                 node_feature_dim,
                 edge_feature_dim_in,
                 edge_hidden_dim,
                 edge_feature_dim_out
                 ):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_dim + edge_feature_dim_in, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_feature_dim_out),
        )

    def forward(self, src, dest, edge_attr, u=None, batch=None):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        return out

class EdgeConvLayerWithoutEdge(nn.Module):

    def __init__(self,
                 node_feature_dim,
                 edge_hidden_dim,
                 edge_feature_dim_out
                 ):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_feature_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, edge_feature_dim_out),
        )

    def forward(self, src, dest, u=None, batch=None):
        out = torch.cat([src, dest], 1)
        out = self.edge_mlp(out)
        return out

class KdModel_lin(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 **kwargs
                 ):
        super().__init__()

        self.lin = Sequential(Linear(node_feature_dim, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = global_mean_pool(x, batch)

        return self.lin(x)


class KdModel_mlp(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 **kwargs
                 ):
        super().__init__()

        self.mlp = Sequential(Linear(node_feature_dim, node_feature_dim // 2), ReLU(),
                              Linear(node_feature_dim // 2, node_embedding_dim), ReLU(), Linear(node_embedding_dim, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = global_mean_pool(x, batch)

        return self.mlp(x)


class KdModel_GCNConv_lin(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 num_layers: int = 4,
                 **kwargs
                 ):

        super().__init__()

        self.convs = ModuleList()
        self.convs.append(
            GCNConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    **kwargs))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(in_channels=node_embedding_dim,
                                      out_channels=node_embedding_dim,
                                      **kwargs))

        self.lin = Sequential(Linear(node_embedding_dim, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv in self.convs:
            h = F.relu(conv(x, edge_index))
            x = h

        x = global_mean_pool(x, batch)
        return self.lin(x)


class KdModel_GCNConv_mlp(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 num_layers: int = 4,
                 **kwargs,
                 ):

        super().__init__()

        self.num_layers = num_layers
        self.convs = ModuleList()

        self.convs.append(
            GCNConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    **kwargs))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(in_channels=node_embedding_dim,
                                      out_channels=node_embedding_dim,
                                      **kwargs))

        self.mlp = Sequential(Linear(node_embedding_dim, node_embedding_dim), ReLU(),
                              Linear(node_embedding_dim, node_embedding_dim // 2), ReLU(),
                              Linear(node_embedding_dim // 2, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv in self.convs:
            h = F.relu(conv(x, edge_index))
            x = h

        x = global_mean_pool(x, batch)
        return self.mlp(x)


class KdModel_GATConv(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 num_layers: int = 4,
                 **kwargs
                 ):

        super().__init__()

        self.convs = ModuleList()

        self.convs.append(
            GATConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    edge_dim=node_embedding_dim,
                    **kwargs))

        for _ in range(num_layers - 1):
            self.convs.append(GATConv(in_channels=node_embedding_dim,
                                      out_channels=node_embedding_dim,
                                      edge_dim=node_embedding_dim,
                                      **kwargs))

        self.mlp = Sequential(Linear(node_embedding_dim, node_embedding_dim), ReLU(),
                              Linear(node_embedding_dim, node_embedding_dim // 2), ReLU(),
                              Linear(node_embedding_dim // 2, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for conv in self.convs:
            h = F.relu(conv(x, edge_index))
            x = h

        x = global_mean_pool(x, batch)
        return self.mlp(x)


class KdModel(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 edge_feature_dim: int,
                 num_layers: int = 4,
                 linear_layer_nodes: bool = False,
                 linear_layer_edges: bool = False,
                 batchnorm: bool = True,
                 **kwargs
                 ):

        super().__init__()

        self.num_layers = num_layers
        self.convs = ModuleList()

        self.linear_layer_nodes = ModuleList() if linear_layer_nodes else None
        self.linear_layer_edges = ModuleList() if linear_layer_edges else None
        self.batchnorms = ModuleList() if batchnorm else None

        self.convs.append(MetaLayer(
            EdgeConvLayer(node_feature_dim=node_feature_dim,
                          edge_feature_dim_in=edge_feature_dim,
                          edge_hidden_dim=node_embedding_dim,
                          edge_feature_dim_out=node_embedding_dim),
            GATConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    edge_dim=node_embedding_dim,
                    **kwargs)))

        for _ in range(self.num_layers - 1):

            meta = MetaLayer(
                EdgeConvLayer(node_feature_dim=node_embedding_dim,
                              edge_feature_dim_in=node_embedding_dim,
                              edge_hidden_dim=node_embedding_dim,
                              edge_feature_dim_out=node_embedding_dim),
                GATConv(in_channels=node_embedding_dim,
                        out_channels=node_embedding_dim,
                        edge_dim=node_embedding_dim,
                        **kwargs))

            self.convs.append(meta)
            if linear_layer_nodes:
                self.linear_layer_nodes.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if linear_layer_edges:
                self.linear_layer_edges.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if batchnorm:
                self.batchnorms.append(BatchNorm(node_embedding_dim))

        self.mlp = Sequential(Linear(node_embedding_dim, node_embedding_dim), ReLU(),
                              Linear(node_embedding_dim, node_embedding_dim // 2), ReLU(),
                              Linear(node_embedding_dim // 2, 1))

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        layers = zip(self.convs, self.batchnorms) if self.batchnorms else self.convs

        for i, layer in enumerate(layers):
            if self.batchnorms:
                conv, batchnorm = layer
            else:
                conv = layer

            h, edge_attr, _ = conv(x, edge_index, edge_attr=edge_attr)
            h = F.relu(batchnorm(h)) if self.batchnorms else F.relu(h)

            if self.linear_layer_nodes:
                if i < self.num_layers - 1:
                    h = F.relu(self.linear_layer_nodes[i](h))

            if self.linear_layer_edges:
                if i < self.num_layers - 1:
                    edge_attr = F.relu(self.linear_layer_edges[i](edge_attr))

            x = h

        x = global_mean_pool(x, batch)
        return self.mlp(x)


class KdModel_PoolEdges(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 edge_feature_dim: int,
                 num_layers: int = 4,
                 linear_layer_nodes: bool = False,
                 linear_layer_edges: bool = False,
                 batchnorm: bool = True,
                 **kwargs
                 ):

        super().__init__()

        self.num_layers = num_layers
        self.convs = ModuleList()

        self.linear_layer_nodes = ModuleList() if linear_layer_nodes else None
        self.linear_layer_edges = ModuleList() if linear_layer_edges else None
        self.batchnorms = ModuleList() if batchnorm else None

        self.convs.append(MetaLayer(
            EdgeConvLayer(node_feature_dim=node_feature_dim,
                          edge_feature_dim_in=edge_feature_dim,
                          edge_hidden_dim=node_embedding_dim,
                          edge_feature_dim_out=node_embedding_dim),
            GATConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    edge_dim=node_embedding_dim,
                    **kwargs)))

        for _ in range(self.num_layers - 1):

            meta = MetaLayer(
                EdgeConvLayer(node_feature_dim=node_embedding_dim,
                              edge_feature_dim_in=node_embedding_dim,
                              edge_hidden_dim=node_embedding_dim,
                              edge_feature_dim_out=node_embedding_dim),
                GATConv(in_channels=node_embedding_dim,
                        out_channels=node_embedding_dim,
                        edge_dim=node_embedding_dim,
                        **kwargs))

            self.convs.append(meta)
            if linear_layer_nodes:
                self.linear_layer_nodes.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if linear_layer_edges:
                self.linear_layer_edges.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if batchnorm:
                self.batchnorms.append(BatchNorm(node_embedding_dim))

        self.mlp = Sequential(Linear(node_embedding_dim, node_embedding_dim), ReLU(),
                              Linear(node_embedding_dim, node_embedding_dim // 2), ReLU(),
                              Linear(node_embedding_dim // 2, 1))

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        layers = zip(self.convs, self.batchnorms) if self.batchnorms else self.convs

        for i, layer in enumerate(layers):
            if self.batchnorms:
                conv, batchnorm = layer
            else:
                conv = layer

            h, edge_attr, _ = conv(x, edge_index, edge_attr=edge_attr)
            h = F.relu(batchnorm(h)) if self.batchnorms else F.relu(h)

            if self.linear_layer_nodes:
                if i < self.num_layers - 1:
                    h = F.relu(self.linear_layer_nodes[i](h))

            if self.linear_layer_edges:
                if i < self.num_layers - 1:
                    edge_attr = F.relu(self.linear_layer_edges[i](edge_attr))

            x = h

        batch_edge = batch[edge_index[0]]

        edge_attr = global_mean_pool(edge_attr, batch_edge)
        return self.mlp(edge_attr)


class KdModel_PoolEdges_WithoutEdge(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 num_layers: int = 4,
                 linear_layer_nodes: bool = False,
                 linear_layer_edges: bool = False,
                 batchnorm: bool = True,
                 **kwargs
                 ):

        super().__init__()

        self.num_layers = num_layers
        self.convs = ModuleList()

        self.linear_layer_nodes = ModuleList() if linear_layer_nodes else None
        self.linear_layer_edges = ModuleList() if linear_layer_edges else None
        self.batchnorms = ModuleList() if batchnorm else None

        self.convs.append(MetaLayer(
            EdgeConvLayerWithoutEdge(node_feature_dim=node_feature_dim,
                                     edge_hidden_dim=node_embedding_dim,
                                     edge_feature_dim_out=node_embedding_dim),
            GATConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    edge_dim=node_embedding_dim,
                    **kwargs)))

        for _ in range(self.num_layers - 1):

            meta = MetaLayer(
                EdgeConvLayerWithoutEdge(node_feature_dim=node_embedding_dim,
                              edge_hidden_dim=node_embedding_dim,
                              edge_feature_dim_out=node_embedding_dim),
                GATConv(in_channels=node_embedding_dim,
                        out_channels=node_embedding_dim,
                        edge_dim=node_embedding_dim,
                        **kwargs))

            self.convs.append(meta)
            if linear_layer_nodes:
                self.linear_layer_nodes.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if linear_layer_edges:
                self.linear_layer_edges.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if batchnorm:
                self.batchnorms.append(BatchNorm(node_embedding_dim))

        self.mlp = Sequential(Linear(node_embedding_dim, node_embedding_dim), ReLU(),
                              Linear(node_embedding_dim, node_embedding_dim // 2), ReLU(),
                              Linear(node_embedding_dim // 2, 1))

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        layers = zip(self.convs, self.batchnorms) if self.batchnorms else self.convs

        for i, layer in enumerate(layers):
            if self.batchnorms:
                conv, batchnorm = layer
            else:
                conv = layer

            h, edge_attr, _ = conv(x, edge_index, edge_attr=edge_attr)
            h = F.relu(batchnorm(h)) if self.batchnorms else F.relu(h)

            if self.linear_layer_nodes:
                if i < self.num_layers - 1:
                    h = F.relu(self.linear_layer_nodes[i](h))

            if self.linear_layer_edges:
                if i < self.num_layers - 1:
                    edge_attr = F.relu(self.linear_layer_edges[i](edge_attr))

            x = h

        batch_edge = batch[edge_index[0]]

        edge_attr = global_mean_pool(edge_attr, batch_edge)
        return self.mlp(edge_attr)

class KdModel_DifferentSizeEdgeConv(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 edge_feature_dim: int,
                 edge_hidden_dim: int,
                 num_layers: int = 4,
                 linear_layer_nodes: bool = False,
                 linear_layer_edges: bool = False,
                 batchnorm: bool = True,
                 **kwargs
                 ):

        super().__init__()

        self.num_layers = num_layers
        self.convs = ModuleList()

        self.linear_layer_nodes = ModuleList() if linear_layer_nodes else None
        self.linear_layer_edges = ModuleList() if linear_layer_edges else None
        self.batchnorms = ModuleList() if batchnorm else None

        self.convs.append(MetaLayer(
            EdgeConvLayer(node_feature_dim=node_feature_dim,
                          edge_feature_dim_in=1,
                          edge_hidden_dim=edge_hidden_dim,
                          edge_feature_dim_out=edge_feature_dim),
            GATConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    edge_dim=edge_feature_dim,
                    **kwargs)))

        for _ in range(self.num_layers - 1):

            meta = MetaLayer(
                EdgeConvLayer(node_feature_dim=node_feature_dim,
                              edge_feature_dim_in=edge_feature_dim,
                              edge_hidden_dim=edge_hidden_dim,
                              edge_feature_dim_out=edge_feature_dim),
                GATConv(in_channels=node_embedding_dim,
                        out_channels=node_embedding_dim,
                        edge_dim=edge_feature_dim,
                        **kwargs))

            self.convs.append(meta)
            if linear_layer_nodes:
                self.linear_layer_nodes.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if linear_layer_edges:
                self.linear_layer_edges.append(nn.Linear(edge_feature_dim, edge_feature_dim))

            if batchnorm:
                self.batchnorms.append(BatchNorm(node_embedding_dim))

        self.mlp = Sequential(Linear(node_embedding_dim, node_embedding_dim), ReLU(),
                              Linear(node_embedding_dim, node_embedding_dim // 2), ReLU(),
                              Linear(node_embedding_dim // 2, 1))

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        layers = zip(self.convs, self.batchnorms) if self.batchnorms else self.convs

        for i, layer in enumerate(layers):
            if self.batchnorms:
                conv, batchnorm = layer
            else:
                conv = layer

            h, edge_attr, _ = conv(x, edge_index, edge_attr=edge_attr)
            h = F.relu(batchnorm(h)) if self.batchnorms else F.relu(h)

            if self.linear_layer_nodes:
                if i < self.num_layers - 1:
                    h = F.relu(self.linear_layer_nodes[i](h))

            if self.linear_layer_edges:
                if i < self.num_layers - 1:
                    edge_attr = F.relu(self.linear_layer_edges[i](edge_attr))

            x = h

        x = global_mean_pool(x, batch)
        return self.mlp(x)

class KdModel_MLPIn(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 edge_feature_dim: int,
                 num_layers: int = 4,
                 linear_layer_nodes: bool = False,
                 linear_layer_edges: bool = False,
                 batchnorm: bool = True,
                 **kwargs
                 ):

        super().__init__()

        self.num_layers = num_layers
        self.convs = ModuleList()

        self.linear_layer_nodes = ModuleList() if linear_layer_nodes else None
        self.linear_layer_edges = ModuleList() if linear_layer_edges else None
        self.batchnorms = ModuleList() if batchnorm else None

        self.convs.append(MetaLayer(
            EdgeConvLayer(node_feature_dim=node_feature_dim,
                          edge_feature_dim_in=node_embedding_dim,
                          edge_hidden_dim=node_embedding_dim,
                          edge_feature_dim_out=node_embedding_dim),
            GATConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    edge_dim=node_embedding_dim,
                    **kwargs)))

        for _ in range(self.num_layers - 1):

            meta = MetaLayer(
                EdgeConvLayer(node_feature_dim=node_embedding_dim,
                              edge_feature_dim_in=node_embedding_dim,
                              edge_hidden_dim=node_embedding_dim,
                              edge_feature_dim_out=node_embedding_dim),
                GATConv(in_channels=node_embedding_dim,
                        out_channels=node_embedding_dim,
                        edge_dim=node_embedding_dim,
                        **kwargs))

            self.convs.append(meta)
            if linear_layer_nodes:
                self.linear_layer_nodes.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if linear_layer_edges:
                self.linear_layer_edges.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if batchnorm:
                self.batchnorms.append(BatchNorm(node_embedding_dim))

        self.mlp_out = Sequential(Linear(node_embedding_dim, node_embedding_dim), ReLU(),
                                  Linear(node_embedding_dim, node_embedding_dim // 2), ReLU(),
                                  Linear(node_embedding_dim // 2, 1))

        self.mlp_in = Sequential(Linear(1, node_embedding_dim// 2 // 2), ReLU(),
                                 Linear(node_embedding_dim// 2 // 2, node_embedding_dim // 2), ReLU(),
                                 Linear(node_embedding_dim // 2, node_embedding_dim))

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        layers = zip(self.convs, self.batchnorms) if self.batchnorms else self.convs

        edge_attr = self.mlp_in(edge_attr)

        for i, layer in enumerate(layers):
            if self.batchnorms:
                conv, batchnorm = layer
            else:
                conv = layer

            h, edge_attr, _ = conv(x, edge_index, edge_attr=edge_attr)
            h = F.relu(batchnorm(h)) if self.batchnorms else F.relu(h)

            if self.linear_layer_nodes:
                if i < self.num_layers - 1:
                    h = F.relu(self.linear_layer_nodes[i](h))

            if self.linear_layer_edges:
                if i < self.num_layers - 1:
                    edge_attr = F.relu(self.linear_layer_edges[i](edge_attr))

            x = h

        x = global_mean_pool(x, batch)
        return self.mlp(x)

class KdModel_DifferentSizeEdgeConvFromEdges(BaseModel):
    def __init__(self,
                 node_feature_dim: Union[int, Tuple[int, int]],
                 node_embedding_dim: int,
                 edge_feature_dim: int,
                 edge_hidden_dim: int,
                 num_layers: int = 4,
                 linear_layer_nodes: bool = False,
                 linear_layer_edges: bool = False,
                 batchnorm: bool = True,
                 **kwargs
                 ):

        super().__init__()

        self.num_layers = num_layers
        self.convs = ModuleList()

        self.linear_layer_nodes = ModuleList() if linear_layer_nodes else None
        self.linear_layer_edges = ModuleList() if linear_layer_edges else None
        self.batchnorms = ModuleList() if batchnorm else None

        self.convs.append(MetaLayer(
            EdgeConvLayer(node_feature_dim=node_feature_dim,
                          edge_feature_dim_in=1,
                          edge_hidden_dim=edge_hidden_dim,
                          edge_feature_dim_out=edge_feature_dim),
            GATConv(in_channels=node_feature_dim,
                    out_channels=node_embedding_dim,
                    edge_dim=edge_feature_dim,
                    **kwargs)))

        for _ in range(self.num_layers - 1):

            meta = MetaLayer(
                EdgeConvLayer(node_feature_dim=node_embedding_dim,
                              edge_feature_dim_in=edge_feature_dim,
                              edge_hidden_dim=edge_hidden_dim,
                              edge_feature_dim_out=edge_feature_dim),
                GATConv(in_channels=node_embedding_dim,
                        out_channels=node_embedding_dim,
                        edge_dim=edge_feature_dim,
                        **kwargs))

            self.convs.append(meta)
            if linear_layer_nodes:
                self.linear_layer_nodes.append(nn.Linear(node_embedding_dim, node_embedding_dim))

            if linear_layer_edges:
                self.linear_layer_edges.append(nn.Linear(edge_feature_dim, edge_feature_dim))

            if batchnorm:
                self.batchnorms.append(BatchNorm(node_embedding_dim))

        self.mlp = Sequential(Linear(edge_feature_dim, edge_feature_dim), ReLU(),
                              Linear(edge_feature_dim, edge_feature_dim // 2), ReLU(),
                              Linear(edge_feature_dim // 2, 1))

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        layers = zip(self.convs, self.batchnorms) if self.batchnorms else self.convs

        for i, layer in enumerate(layers):
            if self.batchnorms:
                conv, batchnorm = layer
            else:
                conv = layer

            h, edge_attr, _ = conv(x, edge_index, edge_attr=edge_attr)
            h = F.relu(batchnorm(h)) if self.batchnorms else F.relu(h)

            if self.linear_layer_nodes:
                if i < self.num_layers - 1:
                    h = F.relu(self.linear_layer_nodes[i](h))

            if self.linear_layer_edges:
                if i < self.num_layers - 1:
                    edge_attr = F.relu(self.linear_layer_edges[i](edge_attr))

            x = h


        batch_edge = batch[edge_index[0]]

        edge_attr = global_mean_pool(edge_attr, batch_edge)
        return self.mlp(edge_attr)
