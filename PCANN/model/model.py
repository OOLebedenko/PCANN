import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from typing import Tuple, Union

from torch.nn import ModuleList, Sequential, ReLU, Linear

from torch_geometric.nn import GATConv, MetaLayer
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
