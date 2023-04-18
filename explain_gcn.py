import os.path as osp

import torch
from torch_geometric.datasets import TUDataset
from models import GCN

from torch_geometric.utils import k_hop_subgraph, to_networkx
import networkx as nx

# import matplotlib.pyplot as plt
from explainer import HeterExplainer
import numpy as np

import os.path as osp
# import argparse
# from torch_geometric.loader import DataLoader

dataset_name = 'MUTAG'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'TU')
# dataset = TUDataset(path, name='MUTAG').shuffle()
dataset = TUDataset(path, name=dataset_name)
ckpt_name = '_'.join((dataset_name, 'GCN'))
ckpt_path = osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', ckpt_name+'.pt')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_features, dataset.num_classes).to(device)

checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['net'])


test_case = [i for i in range(len(dataset)) if dataset[i].y.item()==1]

target = test_case[53]
# target = 171
if target not in test_case:
    raise ValueError('target not in test case.')
data = dataset[target]
# g = to_networkx(data, to_undirected=True)

explainer = HeterExplainer(model, dataset_name, MUTAG_dataset=dataset, device=device)

num_samples = 1000
p_threshold = .05
p_perturb = 0.91
pred_threshold = 0.09
n_cat_value = 3
# delta = 0.01
k = 10
# # check CG
# num_nodes = []
# for target in test_case:
#     num_nodes.append(dataset[target].num_nodes)
# print(round(sum(num_nodes)/len(num_nodes), 2))
# sss
# # check NumRV
# num_RV = []
# for target in test_case:
#     num_RV.append(explainer.getNumRV(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold))
# print(round(sum(num_RV)/len(num_RV),2))
# sss
# # check sample size
# for k in range(10,31,5):
#     print(k)
#     sample_size = []
#     for target in test_case:
#         sample_size.append(explainer.getSampleSize(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold))
#     print(sum(sample_size)/len(sample_size))
# sss
# g = to_networkx(MUTAG_dataset[target], to_undirected=True)
# explainer.orig_pred, explainer.orig_pred_label = explainer.getOrigPred(target)
# print(explainer.getOrigPred(target))
# explainer.uniformPerturbMUTAG(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
# explainer.MLEsamplingMUTAG(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
# blanket_basket = explainer.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold,pred_threshold=pred_threshold)
S, raw_feature_exp, feature_exp, time_used = explainer.explainNoBasket(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
# S, raw_feature_exp, feature_exp, time_used = explainer.explainNotConnected(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

S, raw_feature_exp, feature_exp, time_used = explainer.explain(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
print(explainer.homoCalFidelity(target, S, feature_exp, evaluator=False))

print(f"max percentage: {np.count_nonzero(explainer.cat_y_cap==2)/explainer.cat_y_cap.shape[0]}")
print(f"min percentage: {np.count_nonzero(explainer.cat_y_cap==0)/explainer.cat_y_cap.shape[0]}")
print(explainer.ori_sampled_y_cap.max())
print(explainer.ori_sampled_y_cap.min())

factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=0.0001)
print(explainer.homoCalFidelity(target, factual_S, factual_feat_exp))
print(f"max percentage: {np.count_nonzero(explainer.MLE_cat_y_cap==2)/explainer.MLE_cat_y_cap.shape[0]}")
print(f"min percentage: {np.count_nonzero(explainer.MLE_cat_y_cap==0)/explainer.MLE_cat_y_cap.shape[0]}")
print(explainer.MLE_ori_sampled_y_cap.max())

counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=0.0001)
print(f"max percentage: {np.count_nonzero(explainer.MLE_cat_y_cap==2)/explainer.MLE_cat_y_cap.shape[0]}")
print(f"min percentage: {np.count_nonzero(explainer.MLE_cat_y_cap==0)/explainer.MLE_cat_y_cap.shape[0]}")
print(explainer.homoCalFidelity(target, counterfactual_S, counterfactual_feat_exp))
print(explainer.MLE_ori_sampled_y_cap.min())


# ### entire population: [25, 49]
# f_result = []
# cf_result = []
# target = 25
# factual_feat_exp = {0: [0], 2: [0], 3: [0], 5: [0], 6: [1]}
# counterfactual_feat_exp = {3: [0], 6: [1]}
# f_result.append([x for x in explainer.homoCalFidelity(target, list(factual_feat_exp.keys()), {k:np.array(v) for k, v in factual_feat_exp.items()})[:-1]])
# print(explainer.homoCalFidelity(target, list(factual_feat_exp.keys()), {k:np.array(v) for k, v in factual_feat_exp.items()}))
# cf_result.append([x for x in explainer.homoCalFidelity(target, list(counterfactual_feat_exp.keys()), {k:np.array(v) for k, v in counterfactual_feat_exp.items()})[:-1]])
# print(explainer.homoCalFidelity(target, list(counterfactual_feat_exp.keys()), {k:np.array(v) for k, v in counterfactual_feat_exp.items()}))
# target = 49
# factual_feat_exp = {0: [0], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0], 11: [0], 12: [0]}
# counterfactual_feat_exp = {0: [0], 1: [0], 4: [0], 6: [0]}
# f_result.append([x for x in explainer.homoCalFidelity(target, list(factual_feat_exp.keys()), {k:np.array(v) for k, v in factual_feat_exp.items()})[:-1]])
# print(explainer.homoCalFidelity(target, list(factual_feat_exp.keys()), {k:np.array(v) for k, v in factual_feat_exp.items()}))
# cf_result.append([x for x in explainer.homoCalFidelity(target, list(counterfactual_feat_exp.keys()), {k:np.array(v) for k, v in counterfactual_feat_exp.items()})[:-1]])
# print(explainer.homoCalFidelity(target, list(counterfactual_feat_exp.keys()), {k:np.array(v) for k, v in counterfactual_feat_exp.items()}))
#
# print(np.mean(np.array(f_result), axis=0))
# print(np.mean(np.array(cf_result), axis=0))
# # not connected:time {"25": 7.1508026123046875}{"49": 8.299947738647461}
# [ 0.122 -0.21   0.835  0.041  0.288  0.143]
# [0.041 0.109 0.516 0.023 0.163 0.143]
# ### end of entire population: [25, 49]

entireP = {25: {7: [0], 6: [0], 8: [0], 5: [0], 0: [0], 1: [0], 2: [0], 3: [0], 9: [0], 10: [0], 11: [0], 4: [0], 12: [0]}, 49: {0: [0], 1: [0], 2: [0], 3: [0], 11: [0], 13: [0], 12: [0], 4: [0], 5: [0], 7: [0], 8: [0], 9: [0], 10: [0], 14: [0], 6: [0]}}
# f_result = []
cf_result = []
for target in entireP:
    raw_feature_exp = entireP[target]
    S = list(raw_feature_exp.keys())
    # factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=0.0001)
    counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=0.0001)
    # f_result.append([x for x in explainer.homoCalFidelity(target, list(factual_feat_exp.keys()), {k:np.array(v) for k, v in factual_feat_exp.items()})[:-1]])
    cf_result.append([x for x in explainer.homoCalFidelity(target, list(counterfactual_feat_exp.keys()), {k:np.array(v) for k, v in counterfactual_feat_exp.items()})[:-1]])
    # print(np.round(np.mean(np.array(f_result), axis=0), 4))
    print(np.round(np.mean(np.array(cf_result), axis=0),4))

# graph_id = 171
# x, edge_index = MUTAG_dataset[graph_id].x, MUTAG_dataset[graph_id].edge_index
#
# data = MUTAG_dataset[graph_id]
# # data = dataset[171]
# g = to_networkx(data)
# # nx.draw(g)

# Edge labels:
#   0  aromatic
#   1  single
#   2  double
#   3  triple

# exp_edge_mask = edge_mask>=.5
# # edge_index[:,exp_edge_mask.nonzero().T[0]]
# test_data = MUTAG_dataset[graph_id]
# test_data.edge_index=edge_index[:,exp_edge_mask.nonzero().T[0]]
# g = to_networkx(test_data)
# # nx.draw(g)
# test_g = g.to_undirected()
# S = max(nx.connected_components(test_g), key=len)
# # x[list(S),:]
# exp = nx.induced_subgraph(test_g,list(S))
# # nx.draw(exp)


### check dblp stat
test_case = [25, 49]

num_RV = []
num_CGnodes = []
for target in test_case:
    num_RV.append(explainer.getNumRV(target))

    num_CGnodes.append(dataset[target].num_nodes)
print(sum(num_RV)/len(num_RV))
print(sum(num_CGnodes)/len(num_CGnodes))


### calculated pruning percentage
mean_ratio = []
for target in test_case:
    explainer.computationalG_x_dict = None
    explainer.computationalG_edge_index_dict = None

    g_orig = to_networkx(explainer.MUTAG_dataset[target], to_undirected=True)

    # explainer.computationalG2device()
    # print(explainer.computationalG_x_dict['author'].device)

    explainer.orig_pred, explainer.orig_pred_label = explainer.getOrigPred(target)

    explainer.uniformPerturbMUTAG(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

    blanket_basket, target_ = explainer.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold,
                                                        pred_threshold=pred_threshold)
    # blanket_basket = {k:np.array(list(range(explainer.sampled_data[k].shape[1]))) for k in explainer.sampled_data}
    blanketRV = sum([blanket_basket[n].size for n in blanket_basket if blanket_basket[n].size!=0])
    numRV = explainer.getNumRV(target)
    pruned_ratio = 1-blanketRV/numRV
    print(f"current one ratio: {pruned_ratio}")
    mean_ratio.append(pruned_ratio)
    print('mean ration:')
    print(round(sum(mean_ratio)/len(mean_ratio),4))
###