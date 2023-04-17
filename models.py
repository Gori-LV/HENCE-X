from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.datasets import DBLP
from torch_geometric.nn import Linear#, HANConv, HGTConv

from customized_hgt_conv import HGTConv
from customized_han_conv import HANConv
from typing import Union, Dict

from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
# from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

# import pandas as pd
#
# import networkx as nx
# from DBLP_adj_list import two_hop_neighborhood
# import matplotlib.pyplot as plt
# import numpy as np
# from explainer import Node_Explainer
# from pgmpy.estimators.CITests import chi_square
#
import copy

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, node_types, metadata):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        # ### original code from pyG example
        # for node_type in node_types:
        #     self.lin_dict[node_type] = Linear(-1, hidden_channels)
        #  ###

        ### bot: fix the Linnear(-1, hidden_channels) error
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(node_types[node_type], hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

        self.__embedding4pgexplainer__ = False # for PGExplainer

    def forward(self, x_dict, edge_index_dict):

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        if self.__embedding4pgexplainer__==True:
            self.__embedding_dict__ = x_dict

        return self.lin(x_dict['author'])


class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8, metadata=None):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=metadata)
        self.lin = nn.Linear(hidden_channels, out_channels)

        self.__embedding4pgexplainer__ = False # for PGExplainer

    def forward(self, x_dict, edge_index_dict):
        out = self.han_conv(x_dict, edge_index_dict)

        if self.__embedding4pgexplainer__==True:
            self.__embedding_dict__ = out

        out = self.lin(out['movie'])
        return out


class GIN_5l(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super(GIN_5l, self).__init__()

        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)



class GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_dim, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 128)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
        self.lin1 = Linear(128, 64)
        self.lin2 = Linear(64, out_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # print(x)
        # print(self.conv1.__explain__)
        # print(self.training)
        return F.log_softmax(x, dim=1)
