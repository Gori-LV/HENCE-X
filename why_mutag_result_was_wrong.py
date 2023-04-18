import json
from os import listdir
from os.path import isfile, join

import torch
from torch_geometric.datasets import TUDataset
from models import GCN

from explainer import HeterExplainer
import numpy as np

import os.path as osp

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
explainer = HeterExplainer(model, dataset_name, MUTAG_dataset=dataset, device=device)


for k in [10, 15, 20, 25, 30]:

    print(f"k = {k}")

    result_path = osp.join(osp.dirname(osp.realpath(__file__)), 'result', dataset_name, 'previous_results', f"k{k}/")

    result_files = [f for f in listdir(result_path) if isfile(join(result_path, f))]

    for file in result_files:

        if '_factual_feature_exp.json' in file:
            print('Factual result read from below file:')
            print(file)
            with open(result_path+file, 'r') as f:
                content = f.readline()
            F_feat_exp = json.loads('{' + content[1:-1].replace('}{', ',') + '}')

        if '_counterfactual_feature_exp.json' in file:
            print('Counteractual result read from below file:')
            print(file)
            with open(result_path+file, 'r') as f:
                content = f.readline()
            CF_feat_exp = json.loads('{' + content[1:-1].replace('}{', ',') + '}')

    old_F_result = []
    corret_F_result = []
    for target in F_feat_exp:
        factual_feat_exp = {int(k):np.array(v) for k, v in F_feat_exp[target].items()}
        factual_S = list(factual_feat_exp.keys())
        _, _, F_effect, O_Dens, T_Dens, F_Dens, _ = explainer.homoCalFidelity(int(target), factual_S, factual_feat_exp)
        corret_F_result.append([F_effect, O_Dens, T_Dens, F_Dens])

        ### this is wrong!
        feature_sparsity = {node:factual_feat_exp[node].size/dataset[int(target)].x[node].count_nonzero().item() for node in factual_feat_exp}
        old_F_Dens = sum(feature_sparsity.values())/len(feature_sparsity)
        old_O_Dens = sum(feature_sparsity.values())/dataset[int(target)].num_nodes
        ### this is wrong!

        old_F_result.append([F_effect, old_O_Dens, T_Dens, old_F_Dens])
        corret_F_result.append([F_effect, O_Dens, T_Dens, F_Dens])

    print('Factual evaluation:')
    print('Old result:')
    print(np.round(np.mean(np.array(old_F_result), axis=0), 4))
    print('Correct result:')
    print(np.round(np.mean(np.array(corret_F_result), axis=0), 4))

    print('Counteractual evaluation:')
    old_CF_result = []
    corret_CF_result = []
    for target in CF_feat_exp:
        counterfactual_feat_exp = {int(k): np.array(v) for k, v in CF_feat_exp[target].items()}
        counterfactual_S = list(counterfactual_feat_exp.keys())
        CF_effect, _, _, O_Dens, T_Dens, CF_Dens, _ = explainer.homoCalFidelity(int(target), counterfactual_S,
                                                                                counterfactual_feat_exp)
        corret_CF_result.append([CF_effect, O_Dens, T_Dens, CF_Dens])
        ### this is wrong!
        feature_sparsity = {
            node: counterfactual_feat_exp[node].size / dataset[int(target)].x[node].count_nonzero().item() for node in
            counterfactual_feat_exp}
        old_CF_Dens = sum(feature_sparsity.values()) / len(feature_sparsity)
        old_O_Dens = sum(feature_sparsity.values()) / dataset[int(target)].num_nodes
        ### this is wrong!
        old_CF_result.append([CF_effect, old_O_Dens, T_Dens, old_CF_Dens])
        corret_CF_result.append([CF_effect, O_Dens, T_Dens, CF_Dens])

    print('Old result:')
    print(np.round(np.mean(np.array(old_CF_result), axis=0), 4))
    print('Correct result:')
    print(np.round(np.mean(np.array(corret_CF_result), axis=0), 4))
