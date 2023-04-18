import os.path as osp

from models import HGT

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import DBLP
from torch_geometric.nn import Linear, HGTConv
import pandas as pd

import networkx as nx
from DBLP_adj_list import two_hop_neighborhood
import matplotlib.pyplot as plt
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
# bad_cases = [3558, 3562, 3563, 3575, 3579, 3580, 3584, 3590, 3591, 3593, 3595, 3598, 3599, 3600, 3602, 3604, 3610, 3616, 3619, 3620, 3622, 3627, 3636, 3638, 3640, 3643, 3647, 3668, 3672, 3682, 3683, 3685, 3702, 3703, 3706, 3717, 3721, 3727, 3741, 3744, 3745, 3748, 3749, 3752, 3758, 3759, 3760, 3765, 3767, 3781, 3782, 3795, 3796, 3804, 3808, 3810, 3811, 3812, 3828, 3832, 3834, 3836, 3837, 3842, 3843, 3844, 3846, 3848, 3849, 3854, 3866, 3870, 3875, 3886, 3888, 3889, 3890, 3893, 3896, 3903, 3904, 3914, 3929, 3936, 3941, 3942, 3947, 3956, 3960, 3963, 3966, 3974, 3979, 4010, 4012, 4023, 4043, 4052, 4054, 4055]
# failing_cases = [3604, 3762, 3841, 3824, 1802]
# target = 2 #1802 #1 3170 #3315 [1305, 2207, 2838] 515
zero_feature_cases = (data['author'].x.sum(dim=-1)==0).nonzero().cpu().numpy().T[0].tolist()
test_case = [i for i in range(data['author'].num_nodes) if i not in zero_feature_cases]

# if target in zero_feature_cases:
#     raise ValueError("target in zero feature cases")

# # mapping, reversed_mapping, g = explainer.__subgraph__(target)
# # explainer.orig_pred, explainer.orig_pred_label = explainer.getOrigPred(target, mapping)
# # explainer.uniformPerturbDBLP(target)
# # blanket_basket = explainer.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold,pred_threshold=pred_threshold)

# # S, raw_feature_exp, feature_exp, time_used = explainer.explainNoBasket(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
# # # S, raw_feature_exp, feature_exp, time_used = explainer.explainNotConnected(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

# S, raw_feature_exp, feature_exp, time_used = explainer.explain(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
# # print(explainer.calFidelity(target, S, feature_exp))

# # print(f"max percentage: {np.count_nonzero(explainer.cat_y_cap==2)/explainer.cat_y_cap.shape[0]}")
# # print(f"min percentage: {np.count_nonzero(explainer.cat_y_cap==0)/explainer.cat_y_cap.shape[0]}")

# # factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
# factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb)
# print(explainer.calFidelity(target, factual_S, factual_feat_exp))

# counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb)
# # counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
# print(explainer.calFidelity(target, counterfactual_S, counterfactual_feat_exp))

# explainer.printMeaningDBLP(S, feature_exp)
# explainer.printMeaningDBLP(factual_S, factual_feat_exp)
# explainer.printMeaningDBLP(counterfactual_S, counterfactual_feat_exp)


# # explainer.MLEsamplingDBLP(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=True)
# # for _ in range(1):
# #     eval_metric = []
# #     # for target in bad_cases[-3:-2]:
# #     # for target in zero_feature_cases:
# #     # for target in [1305, 2207, 2838]:
# #     for target in [2]:
# #         # S, raw_feature_exp, feature_exp, time_used = explainer.explain(target, num_samples=num_samples, delta=delta, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
# #         S, raw_feature_exp, feature_exp, time_used = explainer.collapsedExplain(target, num_samples=num_samples, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
# #         test = explainer.calFidelity(target, S, feature_exp)
# #         print(test)
# #         for i in range(n_cat_value):
# #             print(f"how many label {i}? {np.count_nonzero(explainer.cat_y_cap==(i))}")
# #         eval_metric.append(test[:-1])
# #
# #     # explainer.printMeaningDBLP(S, feature_exp)
# # result = np.array(eval_metric)
# # print(np.mean(result, axis=0))
# #
# # print(f'num_samples={num_samples}, p_threshold={p_threshold}, perturb_p={p_perturb}, pred_threshold={pred_threshold}, n_cat_value={n_cat_value}, k={k}')
# #
# # factual_S, factual_feat_exp, counterfactual_S, counterfactual_feat_exp = explainer.MLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
# #
# # print(explainer.calFidelity(target, S, feature_exp))
# # print(explainer.calFidelity(target, factual_S, factual_feat_exp))
# # print(explainer.calFidelity(target, counterfactual_S, counterfactual_feat_exp))
# #
# # # explainer.printMeaningDBLP(S, feature_exp)
# # # explainer.printMeaningDBLP(factual_S, factual_feat_exp)
# # # explainer.printMeaningDBLP(counterfactual_S, counterfactual_feat_exp)

# label_dict = explainer.readLabelDict()

# with open('data/raw/DBLP/conf_label.txt','r') as f:
#     content = f.readlines()
#     conf_label = [int(l.strip().split('\t')[1]) for l in content]

# bag_of_words_paper = np.load('data/DBLP/bag_of_words_paper.npy', allow_pickle=True)
# bag_of_words_author = np.load('data/DBLP/bag_of_words_author.npy', allow_pickle=True)

# ### looking for intro example
# # for node in S:
# #     if node%10==1:
# #         print(bag_of_words_paper[data['paper'].x[node//10].nonzero()].T[0])
# # bag_of_words_author[data['author'].x[target].nonzero()].T[0]

# unmatched_author = []
# for target in range(data['author'].num_nodes):
#     g = explainer.DBLP_computational_graph(target)
#     pub_conf = [n for n in g.nodes if n%10==3]
#     co_author = [n for n in g.nodes if n%10==0 and n//10!=target]
#     conf_match = np.array([conf_label[conf//10]==label_dict['author'][target][1] for conf in pub_conf])
#     author_match = np.array([label_dict['author'][a//10][1]==label_dict['author'][target][1] for a in co_author])
#     if np.all(1-conf_match) and np.all(1-author_match):
#         unmatched_author.append(target)
#         print(target)

# with torch.no_grad():
#     prob = nn.functional.softmax(model(x_dict, edge_index_dict), dim=-1)
#     pred = model(x_dict, edge_index_dict).argmax(dim=-1)
#     # acc = (pred == data['author'].y).sum() / pred.shape[0]
#     acc = (pred == data['author'].y)


# for target in unmatched_author:
#     g = explainer.DBLP_computational_graph(target)
#     if label_dict['author'][target][1]==0:
#         print(target)
#         print(pred[target])
#         explainer.draw_DBLP(target)

# # explainer.draw_DBLP(3464)
# explainer.draw_DBLP(958)
# # print(bag_of_words_author[data['author'].x[3464].nonzero()])
# print(bag_of_words_author[data['author'].x[958].nonzero()])
# print(bag_of_words_author[data['author'].x[615].nonzero()].T[0])
# print(bag_of_words_paper[data['paper'].x[1051].nonzero()].T[0])
# print(bag_of_words_paper[data['paper'].x[2171].nonzero()].T[0])
# ### end of looking for intro example


# # ### to find examples for surrogate model
# # example_target = (data['author'].x>1).nonzero()[:,0].tolist()
# # for target in example_target:
# #     print(target)
# #     explainer.draw_DBLP(target)
# #     g = explainer.DBLP_computational_graph(target)
# #     print(g.number_of_nodes())
# # target = 2928
# # explainer.draw_DBLP(target)
# # g = explainer.DBLP_computational_graph(target) # neighborhood: (27360, 29971, 93381, 113651, 122261, 126991, 1930, 29280, 14280, 53, 133, 163, 173)
# #
# # for node in g.nodes:
# #     if node%10==1:
# #         print(bag_of_words_paper[data['paper'].x[node//10].nonzero()])
# #
# # target = 2928
# # g = explainer.DBLP_computational_graph(target)
# # explainer.draw_DBLP(target)
# # S = [29280, 113651, 126991, 29971, 1930, 53, 163, 173, 1930, 27360]
# # explainer.draw_DBLP(target, S=S)
# # print(bag_of_words_author[data['author'].x[target].nonzero()])
# # print(bag_of_words_paper[data['paper'].x[11365].nonzero()])
# # print(bag_of_words_paper[data['paper'].x[12699].nonzero()])
# # print(bag_of_words_paper[data['paper'].x[2997].nonzero()])

### calculated pruning percentage
mean_ratio = []
for target in test_case:
    explainer.computationalG_x_dict = None
    explainer.computationalG_edge_index_dict = None
    mapping, _, g_orig = explainer.__subgraph__(target)
    # explainer.computationalG2device()
    # print(explainer.computationalG_x_dict['author'].device)

    explainer.orig_pred, explainer.orig_pred_label = explainer.getOrigPred(target, mapping)
    _, _ = explainer.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

    blanket_basket = explainer.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold,
                                               pred_threshold=pred_threshold)
    blanketRV = sum([blanket_basket[n].size for n in blanket_basket if blanket_basket[n].size!=0])
    numRV = explainer.getNumRV(target)
    pruned_ratio = 1-blanketRV/numRV
    print(f"current one ratio: {pruned_ratio}")
    mean_ratio.append(pruned_ratio)
    print('mean ration:')
    print(round(sum(mean_ratio)/len(mean_ratio),4))
