import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data


class InteractionGraph(T.BaseTransform):
    def __call__(self, data: Data) -> Data:
        chain_mask = data.edge_index < data.chain_break_point
        interchain_indexes = torch.argwhere(chain_mask[0] ^ chain_mask[1]).flatten()

        data.edge_attr = data.edge_attr[interchain_indexes]
        data.edge_index = data.edge_index[:, interchain_indexes]

        return data


class DistCut(T.BaseTransform):

    def __init__(self, interactions_cutoff=5):
        self.interactions_cutoff = interactions_cutoff

    def __call__(self, data: Data) -> Data:
        indices = torch.nonzero(data.edge_attr < self.interactions_cutoff)
        data.edge_attr = data.edge_attr[indices[:, 0]]
        data.edge_index = data.edge_index[:, indices[:, 0]]

        return data


class ExpandEdgeAttr(T.BaseTransform):

    def __call__(self, data: Data) -> Data:
        data.edge_attr = torch.tile(data.edge_attr, (1, data.x.shape[-1]))
        return data


class NormalizeNodeFeatures(T.BaseTransform):

    def __call__(self, data: Data) -> Data:
        data.x = data.x - data.x.min()
        data.x.div_(data.x.sum(dim=-1, keepdim=True).clamp_(min=1.))

        return data
