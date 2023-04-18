import os.path as osp

from models import HGT

import torch
# import torch.nn.functional as F
# from torch_geometric.datasets import DBLP
from util import DBLP
# from torch_geometric.nn import Linear, HGTConv
# import pandas as pd

# import networkx as nx
# from DBLP_adj_list import two_hop_neighborhood
# import matplotlib.pyplot as plt
import numpy as np
from explainer import HeterExplainer
# from pgmpy.estimators.CITests import chi_square
import json
# import copy
# import random
import sys
from datetime import datetime
# from util import Logger
import os

from tqdm import tqdm


from os import listdir
from os.path import isfile, join


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

checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['net'])

explainer = HeterExplainer(model, dataset=dataset_name, x_dict=x_dict, edge_index_dict=edge_index_dict, device=device)

num_samples = 1000
p_threshold = .05
p_perturb = 0.52
pred_threshold = .01
n_cat_value = 3
delta = 0.01
k = 10

noBasket = True
notConnected = not noBasket

n_test_batch = 8
test_batch_size = int(np.ceil(data['author'].num_nodes/n_test_batch))

path = 'result/'+dataset_name+'/k'+str(k)+'/raw_result/'
if not os.path.exists(path):
    os.makedirs(path)
timestamp = datetime.today().strftime('%m%d%H%M')


test_batch = 7
restart = 0
c = min(data['author'].num_nodes, test_batch_size*(test_batch+1))-max(restart, test_batch_size*test_batch)
# total = c

parameters = f'num_samples={num_samples}, p_threshold={p_threshold}, perturb_p={p_perturb}, pred_threshold={pred_threshold}, n_cat_value={n_cat_value}, k={k}'
note = f'HGT, 3 cat val, random seed, synthetic node RV, #test batch: {n_test_batch}, working on batch {test_batch}'

if noBasket:
    n_test_batch = 0
    test_batch = 0
    restart = 0
    note+= ', no basket'
if notConnected:
    n_test_batch = 0
    test_batch = 0
    restart = 0
    note+= ', not connected'
sss
prefix = '_'.join([dataset_name, 'p_ptb', str(p_perturb), 'sigma',str(pred_threshold),'batch', str(test_batch), 'n_batch', str(n_test_batch), timestamp])
# prefix = '_'.join([dataset_name, 'p_ptb', str(p_perturb), 'sigma',str(pred_threshold),'batch', str(test_batch), 'batch_size', str(test_batch_size), timestamp])
# prefix = '_'.join([dataset_name, 'makeup_cases', timestamp])
if noBasket:
    prefix = 'noBasket_'+prefix
if notConnected:
    prefix = 'notConnected_'+prefix

open(path+prefix+'_S.json','a').close()
open(path+prefix+'_feature_exp.json','a').close()
open(path+prefix+'_raw_feature_exp.json','a').close()
open(path+prefix+'_time_used.json','a').close()

# bad_cases = {}

failing_cases = []
patience = 5
if noBasket or notConnected:
    patience = 1
try_times = 0

zero_feature_cases = (data['author'].x.sum(dim=-1)==0).nonzero().cpu().numpy().T[0].tolist()

test_case = [i for i in range(max(restart, test_batch_size*test_batch), min(data['author'].num_nodes, test_batch_size*(test_batch+1))) if i not in zero_feature_cases]

if noBasket or notConnected:
    n_chunk = 45
    test_case = [n_chunk*x for x in range(int(data['author'].num_nodes/n_chunk)) if n_chunk*x < data['author'].num_nodes and n_chunk*x not in zero_feature_cases]
    
for target in tqdm(test_case):


    if noBasket:
        S, raw_feature_exp, feature_exp, time_used = explainer.explainNoBasket(target, num_samples=num_samples, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

    elif notConnected:
        S, raw_feature_exp, feature_exp, time_used = explainer.explainNotConnected(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
    else:
        S, raw_feature_exp, feature_exp, time_used = explainer.explain(target, num_samples=num_samples, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
    
    while try_times < patience and len(S)==0:
        try_times += 1

        if noBasket:
            S, raw_feature_exp, feature_exp, time_used = explainer.explainNoBasket(target, num_samples=num_samples, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

        elif notConnected:
            S, raw_feature_exp, feature_exp, time_used = explainer.explainNotConnected(target, num_samples=num_samples, n_cat_value=n_cat_value, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
        else:
            S, raw_feature_exp, feature_exp, time_used = explainer.explain(target, num_samples=num_samples, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
    try_times = 0

    if len(S)==0:
        print(f"CANNOT FIND EXP FOR {target}")
        failing_cases.append(target)

    with open(path+prefix+'_S.json','a') as f:
        json.dump({target:list(S)}, f)
    with open(path+prefix+'_feature_exp.json','a') as f:
        json.dump({target:{key:feature_exp[key].tolist() for key in feature_exp}}, f)
    with open(path+prefix+'_raw_feature_exp.json','a') as f:
        json.dump({target:{key:raw_feature_exp[key].tolist() for key in raw_feature_exp}}, f)
    with open(path+prefix+'_time_used.json','a') as f:
        json.dump({target:time_used}, f)

    print('=======')
    print(note)
    print(parameters)
    print('=======\n')

if len(failing_cases)>0:
    print('failing cases')
    print(failing_cases)
    with open(path+prefix+'_failing_cases.json','w') as f:
        json.dump(failing_cases, f)

# if len(bad_cases)>0:
#     print('bad_cases')
#     print(bad_cases)
#     with open(path+prefix+'_bad_cases.json','w') as f:
#         json.dump(bad_cases, f)

with open(path+prefix+'_parameters.txt','a') as f:
    f.write(parameters)
    f.write(note)


# ### par
# k = 15
# patience = 6
# ### end of par

# num_samples = 1000
# p_threshold = .05
# p_perturb = 0.49
# pred_threshold = .011
# n_cat_value = 3
# delta = 0.01
# # k = 20
par_print = '_'.join(['','p_ptb', str(p_perturb), 'sigma',str(pred_threshold),''])

# zero_feature_cases = (data['author'].x.sum(dim=-1)==0).nonzero().cpu().numpy().T[0].tolist()

result_path = osp.join(osp.dirname(osp.realpath(__file__)), 'result', dataset_name, f"k{k}", 'raw_result/')
if noBasket:
    result_files = [f for f in listdir(result_path) if isfile(join(result_path, f)) and 'noBasket_' in f]
elif notConnected:
    result_files = [f for f in listdir(result_path) if isfile(join(result_path, f)) and 'notConnected_' in f]
else:
    result_files = [f for f in listdir(result_path) if isfile(join(result_path, f))]


# get S
S_dict = {}
for file in result_files:
    # if '_S.json' in file and 'batch' in file:
    if '_S.json' in file and par_print in file:
        print(file)
        with open(result_path+file,'r') as f:
            content = f.read()
        tmp_dict = json.loads('{' + content[1:-1].replace('}{', ',') + '}')
        S_dict.update({int(k): v for k, v in tmp_dict.items()})
print(set(list(range(4057))).difference(set(S_dict.keys())))
# with open(result_path+f'k{k}_S.json','w') as f:
#     json.dump(S_dict, f)

# with open(result_path+f'k{k}_S.json','r') as f:
#     S_dict = json.load(f)
# S_dict = {int(key):value for key, value in S_dict.items()}

raw_feature_exp_dict = {}
for file in result_files:
    # if '_raw_feature_exp.json' in file and 'batch' in file:
    if '_raw_feature_exp.json' in file and par_print in file:
        print(file)
        with open(result_path+file,'r') as f:
            content = f.read()
        tmp_dict = json.loads('{' + content[1:-1].replace('}{', ',') + '}')
        raw_feature_exp_dict.update({int(k): {int(k_): v_ for k_, v_ in v.items()} for k, v in tmp_dict.items()})
print(set(list(range(4057))).difference(set(raw_feature_exp_dict.keys())))
# with open(result_path+'raw_feature_exp.json','w') as f:
#     json.dump(raw_feature_exp_dict, f)

# with open(result_path+'raw_feature_exp.json','r') as f:
#     raw_feature_exp_dict = json.load(f)
# raw_feature_exp_dict = {int(k): {int(k_): v_ for k_, v_ in v.items()} for k, v in raw_feature_exp_dict.items()}


path = f'result/DBLP/k{k}/'
timestamp = datetime.today().strftime('%m%d%H%M')

parameters = f'num_samples={num_samples}, p_threshold={p_threshold}, perturb_p={p_perturb}, pred_threshold={pred_threshold}, n_cat_value={n_cat_value}, k={k}'

# n_test_batch = 8 #fixed
# test_batch_size = int(np.ceil(data['author'].num_nodes/n_test_batch))

# test_batch = 5
restart = 0
# c = min(data['author'].num_nodes, test_batch_size*(test_batch+1))-max(restart, test_batch_size*test_batch)
note = f'HGT, MLE, ' + dataset_name + f' #test batch: {n_test_batch}, working on batch {test_batch}'

# prefix = '_'.join([dataset_name, 'p_ptb', str(p_perturb), 'sigma',str(pred_threshold),'batch', str(test_batch), 'batch_size', str(test_batch_size), timestamp, 'MLE'])

#
# open(path+prefix+'_S.json','a').close()
# open(path+prefix+'_raw_feature_exp.json','a').close()
open(path+prefix+'_counterfactual_S.json','a').close()
open(path+prefix+'_counterfactual_feature_exp.json','a').close()
open(path+prefix+'_factual_S.json','a').close()
open(path+prefix+'_factual_feature_exp.json','a').close()

failing_cases = []
factual_metric = []
counterfactual_metric = []
mean_fid = 1

test_case = [i for i in range(max(restart, test_batch_size*test_batch), min(data['author'].num_nodes, test_batch_size*(test_batch+1))) if i not in zero_feature_cases]

if noBasket or notConnected:
    n_chunk = 45
    test_case = [n_chunk*x for x in range(int(data['author'].num_nodes/n_chunk)) if n_chunk*x < data['author'].num_nodes and n_chunk*x not in zero_feature_cases]
    
for target in tqdm(test_case):
# for target in range(max(restart, test_batch_size*test_batch), min(data['author'].num_nodes, test_batch_size*(test_batch+1))):
#     c-=1

    if target not in S_dict:
        failing_cases.append(target)
        continue

    # if target in zero_feature_cases:
    #     continue

    S = S_dict[target]
    raw_feature_exp = raw_feature_exp_dict[target]
    explainer.computationalG_x_dict = None
    explainer.computationalG_edge_index_dict = None

    if len(S)!=0:
        best_inf = -1
        step = 0
        # while best_fid<mean_fid and step<patience:
        best_factual_S = []
        while best_inf<0.988 and step<patience:
            step += 1
            print(f" +++ factual exp target {target}, step {step}")

            try:
                factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
                factual_eval = explainer.calFidelity(target, factual_S, factual_feat_exp)
                if factual_eval[2]>best_inf:
                    best_inf = factual_eval[2]
                    best_factual_eval = factual_eval
                    best_factual_S, best_factual_feat_exp= factual_S, factual_feat_exp
                print(factual_eval[:-1])

            except:
                continue


        if len(best_factual_S)>0:
            factual_metric.append(best_factual_eval[:-1])
            with open(path+prefix+'_factual_S.json','a') as f:
                json.dump({target:list(best_factual_S)}, f)
            with open(path+prefix+'_factual_feature_exp.json','a') as f:
                json.dump({target:{key:best_factual_feat_exp[key].tolist() for key in best_factual_feat_exp}}, f)
        else:
            failing_cases.append(target)

        best_fid = -1
        step = 0
        # while best_fid<mean_fid and step<patience:
        best_counterfactual_S = []
        while best_fid<0.98 and step<patience:
            step += 1
            print(f" xxx counterfactual exp target {target}, step {step}")
            try:
                counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, n_cat_value=n_cat_value, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
                counterfactual_eval = explainer.calFidelity(target, counterfactual_S, counterfactual_feat_exp)

                if counterfactual_eval[0]>best_fid:
                    best_fid = counterfactual_eval[0]
                    best_counterfactual_eval = counterfactual_eval
                    best_counterfactual_S, best_counterfactual_feat_exp = counterfactual_S, counterfactual_feat_exp
                print(counterfactual_eval[:-1])

            except:
                continue

        if len(best_counterfactual_S)>0:
            counterfactual_metric.append(best_counterfactual_eval[:-1])

            with open(path+prefix+'_counterfactual_S.json','a') as f:
                json.dump({target:list(best_counterfactual_S)}, f)
            with open(path+prefix+'_counterfactual_feature_exp.json','a') as f:
                json.dump({target:{key:best_counterfactual_feat_exp[key].tolist() for key in best_counterfactual_feat_exp}}, f)
        else:
            failing_cases.append(target)
    else:
        failing_cases.append(target)

    # print(f" >>>>>> {c} instances to test.")
    print(note)
    print(parameters)

    result = np.array(factual_metric)
    print(np.round(np.mean(result, axis=0),3))

    result = np.array(counterfactual_metric)

    mean_fid = np.mean(result, axis=0)[0]

    print(np.round(np.mean(result, axis=0),3))
    print('=======\n')

with open(path+prefix+'_factual_metric.json','w') as f:
    json.dump(factual_metric, f)
with open(path+prefix+'_counterfactual_metric.json','w') as f:
    json.dump(counterfactual_metric, f)

# print('failing cases')
# print(failing_cases)
if len(failing_cases)>0:
    print('failing cases')
    print(failing_cases)
    with open(path+prefix+'_failing_cases.json','w') as f:
        json.dump(failing_cases, f)
