from torch_geometric.nn import GCNConv

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.datasets import DBLP
from torch_geometric.nn import Linear#, HANConv, HGTConv

# from customized_hgt_conv import HGTConv
# from customized_han_conv import HANConv
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
from typing import Union, Dict, Optional, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torch_geometric.typing import NodeType, EdgeType, Metadata, Adj
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset

import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.typing import EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax


def HGTgroup(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using HGT, see `examples/hetero/hgt_dblp.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/hgt_dblp.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        group (string, optional): The aggregation scheme to use for grouping
            node embeddings generated by different relations.
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        group: str = "sum",
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.__explain_hgnn__ = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.group = group

        self.k_lin = torch.nn.ModuleDict()
        self.q_lin = torch.nn.ModuleDict()
        self.v_lin = torch.nn.ModuleDict()
        self.a_lin = torch.nn.ModuleDict()
        self.skip = torch.nn.ParameterDict()
        for node_type, in_channels in self.in_channels.items():
            self.k_lin[node_type] = Linear(in_channels, out_channels)
            self.q_lin[node_type] = Linear(in_channels, out_channels)
            self.v_lin[node_type] = Linear(in_channels, out_channels)
            self.a_lin[node_type] = Linear(out_channels, out_channels)
            self.skip[node_type] = Parameter(torch.Tensor(1))

        self.a_rel = torch.nn.ParameterDict()
        self.m_rel = torch.nn.ParameterDict()
        self.p_rel = torch.nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.p_rel[edge_type] = Parameter(torch.Tensor(heads))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict: (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The ouput node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """

        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Iterate over node-types:
        for node_type, x in x_dict.items():
            k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
            q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
            v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over edge-types:
        for edge_type, edge_index in edge_index_dict.items():

            # print(edge_type)
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)

            a_rel = self.a_rel[edge_type]
            k = (k_dict[src_type].transpose(0, 1) @ a_rel).transpose(1, 0)

            m_rel = self.m_rel[edge_type]
            v = (v_dict[src_type].transpose(0, 1) @ m_rel).transpose(1, 0)

            # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
            out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
                                 rel=self.p_rel[edge_type], edge_type=edge_type, size=None)
            out_dict[dst_type].append(out)

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            out = HGTgroup(outs, self.group)

            if out is None:
                out_dict[node_type] = None
                continue

            out = self.a_lin[node_type](F.gelu(out))
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, rel: Tensor,
                index: Tensor, edge_type: str, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = (q_i * k_j).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        # print(out.size())
        # print(out.view(-1, self.out_channe‘ls).size(-2))
        
        # ### directional edge mask
        # if self.__explain_hgnn__:
        #     out = out * self.edge_mask[tuple(edge_type.split('__'))].view([-1] + [1] * (out.dim() - 1))
        # # print(out.view(-1, self.out_channels).shape)
        # ###
        edge_types = {('author','to','paper'), ('paper','to','conference')}
        if self.__explain_hgnn__:
            e_type = tuple(edge_type.split('__'))
            if e_type in edge_types:
                edge_mask = self.edge_mask[e_type].sigmoid()
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
                # out = out * self.edge_mask[e_type].view([-1] + [1] * (out.dim() - 1))
            else:
                reversed_e_type = e_type[::-1]
                edge_mask = self.edge_mask[reversed_e_type].sigmoid()
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
                # out = out * self.edge_mask[reversed_e_type].view([-1] + [1] * (out.dim() - 1))

        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')

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


def HANgroup(xs: List[Tensor], q: nn.Parameter,
          k_lin: nn.Module) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    else:
        num_edge_types = len(xs)
        out = torch.stack(xs)
        attn_score = (q * torch.tanh(k_lin(out)).mean(1)).sum(-1)
        attn = F.softmax(attn_score, dim=0)
        out = torch.sum(attn.view(num_edge_types, 1, -1) * out, dim=0)
        return out


class HANConv(MessagePassing):
    r"""
    The Heterogenous Graph Attention Operator from the
    `"Heterogenous Graph Attention Network"
    <https://arxiv.org/pdf/1903.07293.pdf>`_ paper.

    .. note::

        For an example of using HANConv, see `examples/hetero/han_imdb.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/han_imdb.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.__explain_hgnn__ = False
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.k_lin = nn.Linear(out_channels, out_channels)
        self.q = nn.Parameter(torch.Tensor(1, out_channels))

        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        self.lin_src = nn.ParameterDict()
        self.lin_dst = nn.ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
            self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)
        self.k_lin.reset_parameters()
        glorot(self.q)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict: (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The ouput node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Iterate over node types:
        for node_type, x_node in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x_node).view(
                -1, H, D)
            out_dict[node_type] = []

        # Iterate over edge types:
        for edge_type, edge_index in edge_index_dict.items():
            # print(edge_type)
            src_type, _, dst_type = edge_type
            edge_type = '__'.join(edge_type)
            lin_src = self.lin_src[edge_type]
            lin_dst = self.lin_dst[edge_type]
            x_dst = x_node_dict[dst_type]
            alpha_src = (x_node_dict[src_type] * lin_src).sum(dim=-1)
            alpha_dst = (x_dst * lin_dst).sum(dim=-1)
            alpha = (alpha_src, alpha_dst)
            # propagate_type: (x_dst: Tensor, alpha: PairTensor)
            out = self.propagate(edge_index, x_dst=x_dst, alpha=alpha, edge_type=edge_type,
                                 size=None)

            out = F.relu(out)
            out_dict[dst_type].append(out)

        # iterate over node types:
        for node_type, outs in out_dict.items():
            out = HANgroup(outs, self.q, self.k_lin)

            if out is None:
                out_dict[node_type] = None
                continue
            out_dict[node_type] = out

        return out_dict

    def message(self, x_dst_i: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, edge_type: tuple, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        # print(edge_type)

        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_dst_i * alpha.view(-1, self.heads, 1)
        out = out.view(-1, self.out_channels)

        # for gnnexplainer
        if self.__explain_hgnn__:
            e_type = tuple(edge_type.split('__'))
            # print('applying edge mask on ')
            # print(e_type)
            edge_mask = self.edge_mask[e_type].sigmoid()
            out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'heads={self.heads})')

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
