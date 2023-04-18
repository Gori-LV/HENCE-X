import json
from os import listdir
from os.path import isfile, join
import os.path as osp
from models import HGT

import torch
from torch_geometric.datasets import DBLP
import numpy as np
from explainer import HeterExplainer
import json
import sys
from datetime import datetime

### par
dataset_name = 'MUTAG'
k = 15
### end of par

F_mean = []
CF_mean = []
for k in [10, 15, 20, 25, 30]:

    print(f"k = {k}")

    result_path = osp.join(osp.dirname(osp.realpath(__file__)), 'result', dataset_name, 'previous_results', f"k{k}/")

    result_files = [f for f in listdir(result_path) if isfile(join(result_path, f))]

    factual_metrics = []
    counterfactual_metrics = []
    for file in result_files:
        if '_counterfactual_metric.json' in file and 'batch' in file and '999' in file:
            # print(file)
            with open(result_path + file, 'r') as f:
                content = json.load(f)
            counterfactual_metrics.extend(content)

        if '_factual_metric.json' in file and 'batch' in file and '999' in file:
            # print(file)
            with open(result_path + file, 'r') as f:
                content = json.load(f)
            factual_metrics.extend(content)

    counterfactual_metrics = np.array(counterfactual_metrics)
    CF_mean.append(counterfactual_metrics.mean(axis=0).tolist())

    print(np.round(counterfactual_metrics.mean(axis=0), decimals=4))
    print('')

    factual_metrics = np.array(factual_metrics)
    F_mean.append(factual_metrics.mean(axis=0).tolist())

    print(np.round(factual_metrics.mean(axis=0), decimals=4))
    print('')

print(np.round(np.array(F_mean).mean(axis=0), decimals=4))
print(np.round(np.array(CF_mean).mean(axis=0), decimals=4))


# ### for running time evaluation
# for k in [10, 15, 20, 25, 30]:
#     result_path = osp.join(osp.dirname(osp.realpath(__file__)), 'result', dataset_name, f"k{k}/raw_result/")
#     result_files = [f for f in listdir(result_path) if isfile(join(result_path, f))]
#     for file in result_files:
#         if 'time' in file and 'batch' in file and '99' in file:
#             # print(file)
#             with open(result_path+file, 'r') as f:
#                 content = f.readline()
#             tmp_dict = json.loads('{' + content[1:-1].replace('}{', ',') + '}')
#             mean_runtime = round(sum(list(tmp_dict.values()))/len(tmp_dict), 4)
#             print(mean_runtime)

