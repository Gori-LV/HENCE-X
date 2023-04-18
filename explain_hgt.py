import os.path as osp

from models import HGT

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import DBLP
from torch_geometric.nn import Linear, HGTConv
import pandas as pd

import numpy as np
from explainer import HeterExplainer
from pgmpy.estimators.CITests import chi_square
import json
import copy
import random

dataset_name = 'DBLP'
data_path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset_name)
dataset = DBLP(data_path)
data = dataset[0]
print('WARNING: the dataset is re-processed') # reconstructed dataset: author-dim 451, paper-dim 4233
# print(data)

data['conference'].x = torch.ones(data['conference'].num_nodes, 1)

n_types = data.metadata()[0]
n_types.remove('term')
e_types = [edge for edge in data.metadata()[1] if 'term' not in edge]
e_types_to_remove = [edge for edge in data.metadata()[1] if 'term' in edge]
meta = tuple([n_types, e_types])

### fix Linear -1 dim error
node_types = {node_type:data[node_type].x.size(1) for node_type in n_types}

x_dict = data.x_dict
x_dict.pop('term')
edge_index_dict = data.edge_index_dict
for e in e_types_to_remove:
    edge_index_dict.pop(e)

hidden_channels=64
out_channels=4
num_heads=2
num_layers=2

ckpt_name = '_'.join((dataset_name, 'inDim', str(hidden_channels), 'nHead', str(num_heads),'nLayer', str(num_layers)))
ckpt_name+='_noTerm'
ckpt_path = osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', ckpt_name+'.pt')

model = HGT(hidden_channels=hidden_channels, out_channels=out_channels, num_heads=num_heads, num_layers=num_layers, node_types=node_types, metadata = meta)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['net'])

explainer = HeterExplainer(model, dataset_name, x_dict=x_dict, edge_index_dict=edge_index_dict, device=device)

num_samples = 1000
p_threshold = .05
p_perturb = 0.5
pred_threshold = .01
n_cat_value = 3
delta = 0.01
k = 10

zero_feature_cases = (data['author'].x.sum(dim=-1)==0).nonzero().cpu().numpy().T[0].tolist()
test_case = [i for i in range(data['author'].num_nodes) if i not in zero_feature_cases]

# # S, raw_feature_exp, feature_exp, time_used = explainer.explainNoBasket(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
# # # S, raw_feature_exp, feature_exp, time_used = explainer.explainNotConnected(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
target = 958

S, raw_feature_exp, feature_exp, time_used = explainer.explain(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
# print(explainer.calFidelity(target, S, feature_exp))

# print(f"max percentage: {np.count_nonzero(explainer.cat_y_cap==2)/explainer.cat_y_cap.shape[0]}")
# print(f"min percentage: {np.count_nonzero(explainer.cat_y_cap==0)/explainer.cat_y_cap.shape[0]}")

# factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb)
print(explainer.calFidelity(target, factual_S, factual_feat_exp))

counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb)
# counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
print(explainer.calFidelity(target, counterfactual_S, counterfactual_feat_exp))

explainer.printMeaningDBLP(S, feature_exp)
explainer.printMeaningDBLP(factual_S, factual_feat_exp)
explainer.printMeaningDBLP(counterfactual_S, counterfactual_feat_exp)

