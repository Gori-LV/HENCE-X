from typing import Union, Dict, List
import os.path as osp
import torch.nn as nn
import torch
from torch import nn
import torch.nn.functional as F

from util import IMDB

import torch_geometric.transforms as T

from models import HAN
from explainer import HeterExplainer

import numpy as np
import networkx as nx

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/IMDB')
metapaths = [[('movie', 'actor'), ('actor', 'movie')],
             [('movie', 'director'), ('director', 'movie')]]
transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True,
                           drop_unconnected_nodes=True)
dataset = IMDB(path, transform=transform)
data = dataset[0]
print(data)

# bag_of_words = np.load('data/IMDB/bag_of_words_movie.npy', allow_pickle=True)
# plot_keywords = np.load('data/IMDB/movie_plot_keywords.npy')
# verify that the feature map is correct
# i = 1
# print(plot_keywords[i])
# print(bag_of_words[data['movie'].x[i].nonzero()])

dataset_name = 'IMDB'

hidden_channels = 128
out_channels = 3
num_heads = 8
metadata = data.metadata()

ckpt_name = '_'.join((dataset_name, 'hDim', str(hidden_channels), 'nHead', str(num_heads)))
ckpt_path = osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', ckpt_name+'.pt')


model = HAN(in_channels=-1, out_channels=out_channels, hidden_channels=hidden_channels, heads=num_heads, metadata=metadata)

checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['net'])
print('Trained model loaded.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, model = data.to(device), model.to(device)

explainer = HeterExplainer(model, dataset_name, x_dict=data.x_dict, edge_index_dict=data.edge_index_dict, device=device)

num_samples = 1000
p_threshold = .05
p_perturb = 0.5
pred_threshold = .01
n_cat_value = 3
delta = 0.01
k = 10

target = 15
zero_feature_cases = (data['movie'].x.sum(dim=-1)==0).nonzero().cpu().numpy().T[0].tolist()

if target in zero_feature_cases:
    raise ValueError("target in zero feature cases")

S, raw_feature_exp, feature_exp, time_used = explainer.explain(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
print(explainer.calFidelity(target, S, feature_exp))

# print(f"max percentage: {np.count_nonzero(explainer.cat_y_cap==2)/explainer.cat_y_cap.shape[0]}")
# print(f"min percentage: {np.count_nonzero(explainer.cat_y_cap==0)/explainer.cat_y_cap.shape[0]}")

factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=0.0001)
# factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=0.00001)
print(explainer.calFidelity(target, factual_S, factual_feat_exp))
explainer.printMeaningIMDB(factual_S, factual_feat_exp)

# counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=0.0000001)
counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=0.0001)
print(explainer.calFidelity(target, counterfactual_S, counterfactual_feat_exp))
explainer.printMeaningIMDB(counterfactual_S, counterfactual_feat_exp)

# explainer.printMeaningIMDB(S, feature_exp)
# explainer.printMeaningIMDB(factual_S, factual_feat_exp)
# explainer.printMeaningIMDB(counterfactual_S, counterfactual_feat_exp)
