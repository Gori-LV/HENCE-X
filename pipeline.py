from os import listdir
from os.path import isfile, join
import os.path as osp

import argparse

from models import HGT

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *
from torch_geometric.nn import Linear, HGTConv
import pandas as pd

import numpy as np
from explainer import HeterExplainer
from pgmpy.estimators.CITests import chi_square
import json
import copy
import random

from typing import Union, Dict, List
import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.datasets import IMDB

import torch_geometric.transforms as T

from models import HAN
from explainer import HeterExplainer

import numpy as np
import networkx as nx
# import copy
# import random
import sys
from datetime import datetime
import json

import os

from typing import Union, Dict, List
import os.path as osp

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.datasets import IMDB

import torch_geometric.transforms as T

from models import HAN
from explainer import HeterExplainer

import numpy as np
import networkx as nx

import json
from os import listdir
from os.path import isfile, join
import os.path as osp

import torch
import numpy as np
import sys
from datetime import datetime

from tqdm import tqdm


def arg_parse():
    parser=argparse.ArgumentParser(description="Arguments for HENCE-X.")
    parser.add_argument("--dataset", dest="dataset", type=str, help="Dataset to explain, with specific DGNs:\n DBLP - HGT \n IMDB - HAN \n MUTAG - GCN")
    parser.add_argument("--explain-instance", dest="target", type=int, help="Instances (graphs/nodes) to explain.")
    parser.add_argument("--min_samples", dest="num_samples", type=int, help="Minimum number of samples to generate for a target instance.")
    parser.add_argument("--significance", dest="p_threshold", type=float, help="Statistical significance for conditional independence test (G-test).")
    parser.add_argument("--p_perturb", dest="p_perturb", type=float, help="Probability of perturbing a feature.")
    parser.add_argument("--sigma", dest="pred_threshold", type=float, help="Hyperparameter to categorize GNN score (see Equation (2)).")
    parser.add_argument("--k", dest="k", type=int, help="The multiple in determining sample number (see 4.3.1).")
    parser.add_argument("--save", dest="save", type=bool, help="Whether to save the explanations.")

    # TODO: Check argument usage
    parser.set_defaults(
    	dataset='DBLP',
    	target=None,
    	num_samples=1000,
    	p_threshold=.05,
		p_perturb=0.5,
		pred_threshold=.01,
		k=10,
        save=False
    )
    return parser.parse_args()

def main():
    # Load a configuration
    prog_args = arg_parse()

    # if prog_args.dataset.lower()=='imdb' or prog_args.dataset.lower()=='imdb':

    dataset_name = prog_args.dataset.upper()

    x_dict, edge_index_dict, data, dataset = loadDataset(dataset_name)

    if data is not None:
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

    model = loadGNN(dataset_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name=='DBLP' or dataset_name=='IMDB':
        explainer = HeterExplainer(model, dataset_name, x_dict=x_dict, edge_index_dict=edge_index_dict, device=device)
    else:
        explainer = HeterExplainer(model, dataset_name, MUTAG_dataset=dataset, device=device)

    path = 'result/'+dataset_name+'/k'+str(prog_args.k)+'/raw_result/'
    
    if not os.path.exists(path):
        os.makedirs(path)
    timestamp = datetime.today().strftime('%m%d%H%M')
    prefix = '_'.join([dataset_name, 'p_ptb', str(prog_args.p_perturb), 'sigma', str(prog_args.pred_threshold), timestamp])

    open(path+prefix+'_S.json','a').close()
    open(path+prefix+'_feature_exp.json','a').close()
    open(path+prefix+'_raw_feature_exp.json','a').close()
    open(path+prefix+'_time_used.json','a').close()

    if dataset_name=='DBLP':

        zero_feature_cases = (x_dict['author'].sum(dim=-1)==0).nonzero().cpu().numpy().T[0].tolist()
        test_case = [i for i in range(x_dict['author'].shape[0]) if i not in zero_feature_cases]

        if prog_args.target!=None:
            if prog_args.target in test_case:
                raise ValueError("Illegal target: no feature on target")
            else:
                test_case = [prog_args.target]

    elif dataset_name=='IMDB':
        zero_feature_cases = (data['movie'].x.sum(dim=-1)==0).nonzero().cpu().numpy().T[0].tolist()
        test_case = [i for i in range(data['movie'].num_nodes) if i not in zero_feature_cases]

        if prog_args.target!=None:
            if prog_args.target not in test_case:
                raise ValueError("Illegal target: no feature on target")
            else:
                test_case = [prog_args.target]

    elif dataset_name=='MUTAG':
        test_case = [i for i in range(len(dataset)) if dataset[i].y.item()==1]

        if prog_args.target!=None:
            if prog_args.target not in test_case:
                raise ValueError("Illegal target: instance to be explained is non-mutagenic")
            else:
                test_case = [prog_args.target]

    print('Searching for Markov Blanket...')
    for target in tqdm(test_case):
        S, raw_feature_exp, feature_exp, time_used = explainer.explain(target, num_samples=prog_args.num_samples, k=prog_args.k, p_perturb=prog_args.p_perturb, p_threshold=prog_args.p_threshold, pred_threshold=prog_args.pred_threshold)


        with open(path+prefix+'_S.json','a') as f:
            json.dump({target:list(S)}, f)
        with open(path+prefix+'_feature_exp.json','a') as f:
            json.dump({target:{key:feature_exp[key].tolist() for key in feature_exp}}, f)
        with open(path+prefix+'_raw_feature_exp.json','a') as f:
            json.dump({target:{key:raw_feature_exp[key].tolist() for key in raw_feature_exp}}, f)
        with open(path+prefix+'_time_used.json','a') as f:
            json.dump({target:time_used}, f)

    # parameters = f'num_samples={num_samples}, p_threshold={p_threshold}, perturb_p={p_perturb}, pred_threshold={pred_threshold}, n_cat_value={n_cat_value}, k={k}'

    # with open(path+prefix+'_parameters.txt','a') as f:
    #     f.write(parameters)

    par_print = '_'.join(['','p_ptb', str(prog_args.p_perturb), 'sigma',str(prog_args.pred_threshold),''])

    result_path = osp.join(osp.dirname(osp.realpath(__file__)), 'result', dataset_name, f"k{prog_args.k}", 'raw_result/')
    result_files = [f for f in listdir(result_path) if isfile(join(result_path, f))]

    # get S
    S_dict = {}
    for file in result_files:
        if '_S.json' in file and par_print in file and timestamp in file:
            # print(file)
            with open(result_path+file,'r') as f:
                content = f.read()
            tmp_dict = json.loads('{' + content[1:-1].replace('}{', ',') + '}')
            S_dict.update({int(k): v for k, v in tmp_dict.items()})

    raw_feature_exp_dict = {}
    for file in result_files:
        if '_raw_feature_exp.json' in file and par_print in file and timestamp in file:
            # print(file)
            with open(result_path+file,'r') as f:
                content = f.read()
            tmp_dict = json.loads('{' + content[1:-1].replace('}{', ',') + '}')
            raw_feature_exp_dict.update({int(k): {int(k_): v_ for k_, v_ in v.items()} for k, v in tmp_dict.items()})

    path = 'result/'+dataset_name+f'/k{prog_args.k}/'

    open(path+prefix+'_counterfactual_S.json','a').close()
    open(path+prefix+'_counterfactual_feature_exp.json','a').close()
    open(path+prefix+'_factual_S.json','a').close()
    open(path+prefix+'_factual_feature_exp.json','a').close()

    factual_metric = []
    counterfactual_metric = []

    print('Finding factual and counterfactual explanations...')
    for target in tqdm(test_case):
        S = S_dict[target]
        raw_feature_exp = raw_feature_exp_dict[target]
        explainer.computationalG_x_dict = None
        explainer.computationalG_edge_index_dict = None

        print(f" +++ finding factual explanantion for target {target}")

        factual_S, factual_feat_exp = explainer.factual_synMLE(target, S, raw_feature_exp, num_samples=prog_args.num_samples, k=prog_args.k, p_perturb=prog_args.p_perturb)

        if dataset_name=='MUTAG':
            factual_eval = explainer.homoCalFidelity(target, factual_S, factual_feat_exp)
        else:
            factual_eval = explainer.calFidelity(target, factual_S, factual_feat_exp)
        # print(factual_eval[:-1])

        factual_metric.append(factual_eval[:-1])

        with open(path+prefix+'_factual_S.json','a') as f:
            json.dump({target:list(factual_S)}, f)
        with open(path+prefix+'_factual_feature_exp.json','a') as f:
            json.dump({target:{key:factual_feat_exp[key].tolist() for key in factual_feat_exp}}, f)

        print(f" xxx finding counterfactual explanation for target {target}")

        counterfactual_S, counterfactual_feat_exp = explainer.counterfactual_synMLE(target, S, raw_feature_exp, num_samples=prog_args.num_samples, k=prog_args.k, p_perturb=prog_args.p_perturb)

        if dataset_name=='MUTAG':
            counterfactual_eval = explainer.homoCalFidelity(target, counterfactual_S, counterfactual_feat_exp)
        else:
            counterfactual_eval = explainer.calFidelity(target, counterfactual_S, counterfactual_feat_exp)

        counterfactual_metric.append(counterfactual_eval[:-1])

        with open(path+prefix+'_counterfactual_S.json','a') as f:
            json.dump({target:list(counterfactual_S)}, f)
        with open(path+prefix+'_counterfactual_feature_exp.json','a') as f:
            json.dump({target:{key:counterfactual_feat_exp[key].tolist() for key in counterfactual_feat_exp}}, f)

        F_metric = np.round(np.mean(np.array(factual_metric), axis=0),3)
        print('----- Accumulated Mean -----')
        print(f"F^Hence-X\nF-effect: {F_metric[2]} O-Dens: {F_metric[3]}   T-Dens: {F_metric[4]}  F-Dens: {F_metric[5]}")
        # print(np.round(np.mean(result, axis=0),3))

        CF_metric = np.round(np.mean(np.array(counterfactual_metric), axis=0),3)
        print(f"CF^Hence-X\nCF-effect: {CF_metric[0]} O-Dens: {CF_metric[3]}   T-Dens: {CF_metric[4]}  F-Dens: {CF_metric[5]}")
        print('============================\n')

    with open(path+prefix+'_factual_metric.json','w') as f:
        json.dump(factual_metric, f)
    with open(path+prefix+'_counterfactual_metric.json','w') as f:
        json.dump(counterfactual_metric, f)

    
if __name__=="__main__":
	main()