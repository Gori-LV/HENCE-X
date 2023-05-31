import os.path as osp

import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx, sort_edge_index

from scipy import stats
from scipy.special import softmax
from scipy.cluster.hierarchy import fcluster, linkage

from pgmpy.estimators.CITests import chi_square, g_sq
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import math
import time
import copy
import random
import json

class HeterExplainer:
    def __init__(
        self,
        model,
        dataset,
        device,
        x = None,
        edge_index = None,
        MUTAG_dataset = None,
        x_dict = None,
        edge_index_dict = None
    ):
        self.model = model
        self.model.eval()

        self.dataset = dataset
        self.device = device

        if self.dataset=='MUTAG':
            self.MUTAG_dataset = [data.to(device) for data in MUTAG_dataset]
            print("Explainer (homo) set up on "+self.device.type)

        else:
            self.x_dict = x_dict
            self.edge_index_dict = edge_index_dict

            self.node_types = list(x_dict.keys())
            self.edge_types = list(edge_index_dict.keys())

            self.computationalG_x_dict = None
            self.computationalG_edge_index_dict = None

            self.DBLP_idx2node_type = {0:'author', 1:'paper', 3:'conference'}
            # self.DBLP_node_type2idx = {'author':0, 'paper':1, 'term':2, 'conference':3}
            self.DBLP_node_type2idx = {v: k for k, v in self.DBLP_idx2node_type.items()}

            self.IMDB_label2genre = {0:'Action', 1:'Comedy', 2:'Drama'}
            self.DBLP_label2filed = {0: 'Database', 1: 'Data Mining', 2: 'AI', 3: 'Information Retrieval'}
            print("Explainer (heter) set up on "+self.device.type)

    def DBLP_computational_graph(self, target):
        # return a nx graph object with special encode node id
        # For HGT on DBLP (layer=2), give 2-hop neighborhood only, and the default target type is 'author'
        # use zero-padding, keep the computational graph only
        
        g = nx.Graph(id=target)
        g.add_nodes_from([(target*10, {'type':'author', 'target':1})]) # little trick to encode type info in the node id in nx graph. The last digit is the type of the node.
        # author: 0, paper: 1, term: 2, conference: 3, just as in the data.node_types

        # 1-hop neighbors, type: paper
        one_hop_neighbors = self.edge_index_dict[('author', 'to', 'paper')][1][self.edge_index_dict[('author', 'to', 'paper')][0]==target] # data.edge_stores[0] is author-paper
        g.add_nodes_from([(x.item()*10+1, {'type':'paper', 'target':0}) for x in one_hop_neighbors])
        g.add_edges_from([(target*10, x.item()*10+1) for x in one_hop_neighbors])

        # 2-hop neighbors, type: author, conference, term

        # 2-hop author
        two_hop_author = []
        for n in one_hop_neighbors:
            n = n.item()
            neighbors = self.edge_index_dict[('paper','to','author')][1][self.edge_index_dict[('paper','to','author')][0]==n] # data.edge_stores[1] is paper-author (symmetric)
            g.add_nodes_from([(x.item() * 10, {'type': 'author', 'target':0}) for x in neighbors if x.item() * 10 not in g.nodes])
            g.add_edges_from([(n * 10 + 1, x.item() * 10) for x in neighbors])
            two_hop_author += [x.item() for x in neighbors]
        two_hop_author = set(two_hop_author)

        # 2-hop conference
        two_hop_conf = []
        for n in one_hop_neighbors:
            n = n.item()
            neighbors = self.edge_index_dict[('paper','to','conference')][1][self.edge_index_dict[('paper','to','conference')][0]==n] # data.edge_stores[3] is paper-conference
            # print(neighbors)
            g.add_nodes_from([(x.item() * 10+3, {'type': 'conference','target':0}) for x in neighbors if x.item() * 10+3 not in g.nodes])
            g.add_edges_from([(n * 10 + 1, x.item() * 10+3) for x in neighbors])
            two_hop_conf += [x.item() for x in neighbors]
        two_hop_conf = set(two_hop_conf)

        return g

    def IMDB_computational_graph(self, target):
        # return a nx graph object of IMDB dataset for HAN model, all nodes have a type of 'movie', edge type are the two metapaths: MAM, MDM.

        g = nx.Graph(id=target)
        g.add_nodes_from([(target, {'target': 1})])

        for metapath in self.edge_index_dict.keys():

            neighbors = self.edge_index_dict[metapath][1][self.edge_index_dict[metapath][0]==target]

            g.add_nodes_from([(x.item(), {'target': 0}) for x in neighbors])
            g.add_edges_from([(target, x.item())for x in neighbors], label=metapath)

        return g

    def draw_DBLP(self, target, S=None, if_save = True, path = 'result/', name = None):

        self.computationalG_x_dict = None
        self.computationalG_edge_index_dict = None
        
        g = self.DBLP_computational_graph(target)

        if S is None:
            neighbors = g.nodes
            name = 'raw'
        else:
            neighbors = S
            if name is None:
                name = str(target)+'exp'
            g = g.subgraph(neighbors)
            # nx.draw_networkx_nodes(g, pos, nodelist=[x_ for x_ in g.nodes if x_ not in S], node_color="white", node_size=1)

        fsize = (min(7.26, 1 * len(g.nodes)), min(5.5, 1 * len(g.nodes)))
        pos = nx.spring_layout(g, scale=.4)

        plt.figure(figsize=fsize)
        plt.box(False)
        

        label_dict = self.readLabelDict()


        nx.draw_networkx_nodes(g, pos, nodelist=[x_ for x_ in neighbors if x_ % 10 == 1 and g.nodes[x_]['target']!=1], node_color="blue", node_size=250) # draw papers
        nx.draw_networkx_nodes(g, pos, nodelist=[x_ for x_ in neighbors if x_ % 10 == 0 and g.nodes[x_]['target']!=1], node_color="pink", node_size=250) # draw author
        nx.draw_networkx_nodes(g, pos, nodelist=[x_ for x_ in neighbors if x_ % 10 == 3 and g.nodes[x_]['target']!=1], node_color="green", node_size=250) # draw conference
        nx.draw_networkx_nodes(g, pos, nodelist=[x_ for x_ in neighbors if x_ % 10 == 2 and g.nodes[x_]['target']!=1], node_color="yellow", node_size=250) # draw term
        nx.draw_networkx_nodes(g, pos, nodelist=[x_ for x_ in neighbors if x_ % 10 == 0 and g.nodes[x_]['target']==1], node_color="red", node_size=250) # draw target node
        nx.draw_networkx_edges(
            g,
            pos,
            edge_color='gray',
            width=2,
            alpha=1)
        
        if label_dict is None:
            v_lb ={v:g.nodes[v]['type']+str(int(v/10)) for v in g.nodes}
        else:
            labels = ['DB','DM','AI','IR']
            lb_author = {v:label_dict['author'][v//10][-1]+'('+labels[label_dict['author'][v//10][1]]+')' for v in neighbors if v%10==0}

            lb_conf ={v:label_dict['conference'][v//10][-1] for v in neighbors if v%10==3}
            lb_paper = {v:'P'+str(v//10) for v in neighbors if v%10==1}
            lb_term ={v:'"'+str(label_dict[self.DBLP_idx2node_type[v%10]][v//10][-1])+'"' for v in neighbors if v%10==2}

            v_lb = {**lb_paper, **lb_author, **lb_term, **lb_conf}

            content = ''
            for v in [x_ for x_  in neighbors if x_%10 ==1]:
                content+= 'P'+str(v//10)+': '+label_dict['paper'][v//10][-1]+'\n'
            print(content.title())

        nx.draw_networkx_labels(g, pos, v_lb)
        
        if if_save:
            name += str(g.graph['id'])+'.png'
            plt.savefig(path+name,bbox_inches="tight", transparent=True)
        else:
            plt.show()
        # plt.savefig(savefig_path+ str(gnx.graph['id'])+'.png',transparent=True)
        plt.close()


    def __subgraph__(self, target):

        if self.dataset=='DBLP':

            g = self.DBLP_computational_graph(target)

            x_dict = {}
            mapping = {}
            reversed_mapping = {}
            for node_type_idx in self.DBLP_idx2node_type:

                node_list = [n//10 for n in g.nodes if n%10==node_type_idx]
                mapping[self.DBLP_idx2node_type[node_type_idx]]={node_list[i]:i for i in range(len(node_list))}

                reversed_node_list = [n for n in g.nodes if n%10==node_type_idx]
                reversed_mapping[self.DBLP_idx2node_type[node_type_idx]]={i:reversed_node_list[i] for i in range(len(reversed_node_list))}

                x_dict[self.DBLP_idx2node_type[node_type_idx]] = self.x_dict[self.DBLP_idx2node_type[node_type_idx]][node_list, :]

            # print(mapping)

            # edge_types = {('author','to','paper'), ('paper','to','term'),('paper','to','conference')}
            edge_types = {('author','to','paper'), ('paper','to','conference')}
            edge_list = list(g.edges)

            edge_index_dict = {}

            for edge_type in edge_types:

                h_idx = self.DBLP_node_type2idx[edge_type[0]]
                t_idx = self.DBLP_node_type2idx[edge_type[-1]]
                # print(h_idx, t_idx)

                edges = [(mapping[self.DBLP_idx2node_type[h_idx]][e[0]//10], mapping[self.DBLP_idx2node_type[t_idx]][e[1]//10]) for e in edge_list if e[0]%10==h_idx and e[1]%10==t_idx]
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

                # edge_index = sort_edge_index(edge_index, sort_by_row=False)
                edge_index_dict[edge_type] = edge_index

                reverse_type = edge_type[::-1]
                # edge_index_dict[reverse_type] = sort_edge_index(torch.flipud(edge_index), sort_by_row=False)

                edge_index_dict[reverse_type] = torch.flipud(edge_index)

            # if not hasattr(self,'computationalG_x_dict'):
            if self.computationalG_x_dict is None:
                self.computationalG_x_dict = x_dict
            # if not hasattr(self,'computationalG_edge_index_dict'):
            if self.computationalG_edge_index_dict is None:
                self.computationalG_edge_index_dict = edge_index_dict

        elif self.dataset=='IMDB':

            g = self.IMDB_computational_graph(target)

            mapping = {'movie':{v:i for i, v in enumerate(list(g.nodes))}}
            reversed_mapping = {'movie':{v:k for k, v in mapping['movie'].items()}}

            # self.computationalG_edge_index_dict = {}
            # for metapath in self.edge_index_dict.keys():
            #     tmp = self.edge_index_dict[metapath][:, self.edge_index_dict[metapath][0]==target]
            #     self.computationalG_edge_index_dict[metapath] = torch.unique(torch.cat((tmp, torch.flipud(tmp)), dim=-1), dim=-1)
            #     # self.computationalG_edge_index_dict[metapath] = torch.cat((tmp, torch.flipud(tmp)), dim=-1)

            edge_index_dict = {}
            for metapath in self.edge_index_dict.keys():

                edge_list = [e for e in list(g.edges) if g.edges[e]['label']==metapath]
                edges = [(mapping['movie'][target],mapping['movie'][target])] + [(mapping['movie'][e[0]], mapping['movie'][e[1]]) for e in edge_list]
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_index_dict[metapath] = torch.unique(torch.cat((edge_index, torch.flipud(edge_index)), dim=-1), dim=-1)

            if self.computationalG_x_dict is None:
                self.computationalG_x_dict = {'movie': self.x_dict['movie'][list(g.nodes)]}

            if self.computationalG_edge_index_dict is None:
                self.computationalG_edge_index_dict = edge_index_dict
            
        # self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

        return mapping, reversed_mapping, g

    def vec2categ(self, vec):
        if vec.shape[1]==1:
            return vec

        base = np.array([2**k for k in range(vec.shape[1])])
        base = np.expand_dims(base, axis=0)
        base = np.repeat(base, vec.shape[0], axis=0)
        # print(base.shape)
        # cat = np.expand_dims(np.sum(base+vec, axis=-1),axis=0).T
        cat = np.expand_dims(np.sum(base*vec, axis=-1),axis=0).T
        cat = cat - np.min(cat)*np.ones(cat.shape, dtype = np.int8)

        return cat

    def syntheticNodeRV(self, vec):

        vec = np.concatenate((vec, self.cat_y_cap), axis=1)

        base = np.array([10**(vec.shape[1]-k-1) for k in range(vec.shape[1])])
        base = np.expand_dims(base, axis=0)
        base = np.repeat(base, vec.shape[0], axis=0)
        # print(base.shape)
        # cat = np.expand_dims(np.sum(base+vec, axis=-1),axis=0).T
        cat = np.expand_dims(np.sum(base*vec, axis=-1),axis=0).T

        return cat

    @torch.no_grad()
    def getOrigPred(self, target, mapping=None): # for DBLP(IMDB), all testing nodes are authors(movies)
        
        self.model = self.model.to(self.device)
        self.model.eval()


        if self.dataset=='MUTAG':
            data = self.MUTAG_dataset[target].to(self.device)
            out = nn.functional.softmax(self.model(data.x, data.edge_index, torch.tensor([0 for _ in range(data.x.shape[0])]).to(self.device))[0], dim=-1) # model(x, edge_index, batch)

        else:
            self.computationalG2device()

            with torch.no_grad():
                if self.dataset=='DBLP':
                    out = nn.functional.softmax(self.model(self.computationalG_x_dict, self.computationalG_edge_index_dict), dim=-1)[mapping['author'][target]]
                # pred_score, pred_label = out.max().item(), out.argmax().item()
                elif self.dataset=='IMDB':
                    out = nn.functional.softmax(self.model(self.computationalG_x_dict, self.computationalG_edge_index_dict), dim=-1)[mapping['movie'][target]]

        return out.max().item(), out.argmax().item()

    @torch.no_grad()
    def MLEsamplingDBLP(self, target, S=[], raw_feature_exp={}, n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, pred_threshold=0.01, factual=True):
        # for u, do element-wise feature sampling; for S, do categorical sampling based on feature explanation
        
        mapping, reversed_mapping, g = self.__subgraph__(target)
        
        self.MLE_sampling_fail = False

        num_RV = n_cat_value

        for n in S:
            if n not in raw_feature_exp:
                num_RV += 1
            else:
                num_RV += len(raw_feature_exp[n])
            
        num_samples = max(k*num_RV, num_samples)
        # num_samples_needed = k*num_RV
        # if num_samples_needed>=num_samples:
        #     num_samples=num_samples_needed

        # print("Generating "+str(num_samples)+" samples on target: " + str(target))

        self.MLE_sampled_data = {}

        self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

        sampled_y_cap = []

        for iteration in range(num_samples):
            # print('*** Iter: '+str(iteration))

            if factual:
                X_perturb = {k:torch.zeros_like(self.computationalG_x_dict[k]) for k in self.computationalG_x_dict}
            else:
                X_perturb = copy.deepcopy(self.computationalG_x_dict)
            # print('X_perturb')
            # print(X_perturb['author'].device)

            # print('perturbing nodes in S...')
            for n in S:

                # print('perturbing nodes in S: '+str(n))
                node_type = self.DBLP_idx2node_type[n%10]
                node = mapping[node_type][n//10]
                # print(node)

                if node_type=='author' or node_type=='paper':
                # if n in raw_feature_exp and (node_type=='author' or node_type=='paper'):

                    pos = self.computationalG_x_dict[node_type][node].nonzero().cpu().numpy()[raw_feature_exp[n]]
                    # print(pos)

                    perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)

                    tmp = (self.computationalG_x_dict[node_type][node][pos.T]!=perturb_array).cpu().detach().numpy()
                    tmp = np.expand_dims(tmp, axis=0)

                    X_perturb[node_type][node][pos.T] = perturb_array


                elif node_type=='conference':
                # else:
                    seed = random.choices([0,1], weights = [1-p_perturb, p_perturb], k=1)[0]

                    if seed==1:
                        X_perturb[node_type][node] = 0
                        tmp = np.ones((1,1), dtype=np.int8)
                    else:
                        X_perturb[node_type][node] = 1
                        tmp = np.zeros((1,1), dtype=np.int8)

                if n not in self.MLE_sampled_data:
                    self.MLE_sampled_data[n] = tmp
                else:
                    self.MLE_sampled_data[n] = np.append(self.MLE_sampled_data[n], tmp, axis=0)

            if X_perturb['author'].device != self.device:        
                for nt in X_perturb:
                    X_perturb[nt] = X_perturb[nt].to(self.device)
            
            # self.computationalG2device()
            
            with torch.no_grad():
                out = nn.functional.softmax(self.model(X_perturb, self.computationalG_edge_index_dict), dim=-1)

            pred_score= out[mapping['author'][target]][self.orig_pred_label].item()            
            sampled_y_cap.append(pred_score)

        self.MLE_ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T

        perturb_range = self.MLE_ori_sampled_y_cap.max() - self.MLE_ori_sampled_y_cap.min()

        if perturb_range==0: 

            # print('GNN prediction never change!')
            self.MLE_sampling_fail = True
            self.MLE_cat_y_cap = np.ones(self.MLE_ori_sampled_y_cap.shape)
            return g, mapping

        elif perturb_range<pred_threshold:
            # print('perturb range too small, decrease pred_threshold')
            pred_threshold/=2

        self.MLE_cat_y_cap =  np.where(self.MLE_ori_sampled_y_cap<=(self.MLE_ori_sampled_y_cap.min()+pred_threshold), 0, self.MLE_ori_sampled_y_cap)
        self.MLE_cat_y_cap =  np.where(self.MLE_cat_y_cap>=(self.MLE_ori_sampled_y_cap.max()-pred_threshold), 2, self.MLE_cat_y_cap)
        self.MLE_cat_y_cap =  np.where((0<self.MLE_cat_y_cap) & (self.MLE_cat_y_cap<2), 1, self.MLE_cat_y_cap)

        self.MLE_cat_y_cap = self.MLE_cat_y_cap.astype(int)


        if factual:
            how_many_more = 1-np.count_nonzero(self.MLE_cat_y_cap==2)
        else:
            how_many_more = 1-np.count_nonzero(self.MLE_cat_y_cap==0) 

        counts = np.array([np.count_nonzero(self.MLE_cat_y_cap==val) for val in range(3)])

        # print('how many more:')
        # print(how_many_more)

        if how_many_more>0:

            step = 0

            to_substitute = np.argmax(counts)
            p_perturb = self.adjustPperturb(np.argmin(counts))
            # print(f"new p perturb: {p_perturb}")

            while how_many_more>0 and step<2*num_samples:
                # print(how_many_more)

                step+=1

                if factual:
                    X_perturb = {k:torch.zeros_like(self.computationalG_x_dict[k]) for k in self.computationalG_x_dict}
                else:
                    X_perturb = copy.deepcopy(self.computationalG_x_dict)
                # print('X_perturb')
                # print(X_perturb['author'].device)

                # print('perturbing nodes in S...')
                for n in S:

                    # print('perturbing nodes in S: '+str(n))
                    node_type = self.DBLP_idx2node_type[n%10]
                    node = mapping[node_type][n//10]
                    # print(node)

                    if node_type=='author' or node_type=='paper':
                    # if n in raw_feature_exp and (node_type=='author' or node_type=='paper'):

                        pos = self.computationalG_x_dict[node_type][node].nonzero().cpu().numpy()[raw_feature_exp[n]]
                        # print(pos)

                        perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)

                        tmp = (self.computationalG_x_dict[node_type][node][pos.T]!=perturb_array).cpu().detach().numpy()
                        tmp = np.expand_dims(tmp, axis=0)

                        X_perturb[node_type][node][pos.T] = perturb_array


                    elif node_type=='conference':
                    # else:
                        seed = random.choices([0,1], weights = [1-p_perturb, p_perturb], k=1)[0]

                        if seed==1:
                            X_perturb[node_type][node] = 0
                            tmp = np.ones((1,1), dtype=np.int8)
                        else:
                            X_perturb[node_type][node] = 1
                            tmp = np.zeros((1,1), dtype=np.int8)

                    if n not in self.MLE_sampled_data:
                        self.MLE_sampled_data[n] = tmp
                    else:
                        self.MLE_sampled_data[n] = np.append(self.MLE_sampled_data[n], tmp, axis=0)

                if X_perturb['author'].device != self.device:        
                    for nt in X_perturb:
                        X_perturb[nt] = X_perturb[nt].to(self.device)
                
                # self.computationalG2device()
                
                with torch.no_grad():
                    out = nn.functional.softmax(self.model(X_perturb, self.computationalG_edge_index_dict), dim=-1)

                pred_score= out[mapping['author'][target]][self.orig_pred_label].item()  

                if factual and pred_score>=self.MLE_ori_sampled_y_cap.max()-pred_threshold:
                    # print(f"Got one from {2}!")
                    how_many_more-=1
                    # print(np.sum(how_many_more))
                    # print(how_many_more)

                    sub_pos = np.random.choice(np.nonzero(self.MLE_cat_y_cap==1)[0], size=1)

                    self.MLE_cat_y_cap[sub_pos] = 2

                    for key in self.MLE_sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.MLE_sampled_data[key][sub_pos,:]=self.MLE_sampled_data[key][-1:,:]
                        self.MLE_sampled_data[key] = self.MLE_sampled_data[key][:-1, :]   

                elif not factual and pred_score<=self.MLE_ori_sampled_y_cap.min()+pred_threshold:

                    # print(f"Got one from {0}!")
                    how_many_more-=1
                    # print(np.sum(how_many_more))
                    # print(how_many_more)

                    sub_pos = np.random.choice(np.nonzero(self.MLE_cat_y_cap==1)[0], size=1)

                    self.MLE_cat_y_cap[sub_pos] = 0

                    for key in self.MLE_sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.MLE_sampled_data[key][sub_pos,:]=self.MLE_sampled_data[key][-1:,:]
                        self.MLE_sampled_data[key] = self.MLE_sampled_data[key][:-1, :]   
                else:
                    for key in self.MLE_sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.MLE_sampled_data[key] = self.MLE_sampled_data[key][:-1, :]
            if how_many_more>0 and step==10*num_samples:
                self.MLE_sampling_fail = True
                # print('WARNING: MLE sampling failed!')

        return g, mapping

    @torch.no_grad()
    def MLEsamplingIMDB(self, target, S=[], raw_feature_exp={}, n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, pred_threshold=0.01, factual=True):
        # for u, do element-wise feature sampling; for S, do categorical sampling based on feature explanation
        
        mapping, reversed_mapping, g = self.__subgraph__(target)

        self.MLE_sampling_fail = False

        num_RV = n_cat_value

        for n in S:
            if n not in raw_feature_exp:
                num_RV += 1
            else:
                num_RV += len(raw_feature_exp[n])
            
        num_samples = max(k*num_RV, num_samples)
        # num_samples_needed = k*num_RV
        # if num_samples_needed>=num_samples:
        #     num_samples=num_samples_needed

        # print("Generating "+str(num_samples)+" samples on target: " + str(target))

        self.MLE_sampled_data = {}

        self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

        sampled_y_cap = []

        node_type = 'movie'

        for iteration in range(num_samples):
            # print('*** Iter: '+str(iteration))

            if factual:
                X_perturb = {k:torch.zeros_like(self.computationalG_x_dict[k]) for k in self.computationalG_x_dict}
            else:
                X_perturb = copy.deepcopy(self.computationalG_x_dict)
            # print('X_perturb')
            # print(X_perturb['author'].device)

            # print('perturbing nodes in S...')
            for n in S:
                # print('perturbing nodes in S: '+str(n))

                node = mapping[node_type][n]
                # print(node)

                pos = self.computationalG_x_dict[node_type][node].nonzero().cpu().numpy()[raw_feature_exp[n]]
                # print(pos)

                perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)

                tmp = (self.computationalG_x_dict[node_type][node][pos.T]!=perturb_array).cpu().detach().numpy()
                tmp = np.expand_dims(tmp, axis=0)

                X_perturb[node_type][node][pos.T] = perturb_array

                if n not in self.MLE_sampled_data:
                    self.MLE_sampled_data[n] = tmp
                else:
                    self.MLE_sampled_data[n] = np.append(self.MLE_sampled_data[n], tmp, axis=0)

            if X_perturb['movie'].device != self.device:        
                for nt in X_perturb:
                    X_perturb[nt] = X_perturb[nt].to(self.device)
            
            # self.computationalG2device()
            
            with torch.no_grad():
                out = nn.functional.softmax(self.model(X_perturb, self.computationalG_edge_index_dict), dim=-1)

            pred_score= out[mapping['movie'][target]][self.orig_pred_label].item()            
            sampled_y_cap.append(pred_score)

        self.ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T

        self.MLE_ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T

        perturb_range = self.MLE_ori_sampled_y_cap.max() - self.MLE_ori_sampled_y_cap.min()

        if perturb_range==0: 

            # print('GNN prediction never change!')
            self.MLE_sampling_fail = True
            self.MLE_cat_y_cap = np.ones(self.MLE_ori_sampled_y_cap.shape)
            return g, mapping

        elif perturb_range<pred_threshold:
            # print('perturb range too small, decrease pred_threshold')
            pred_threshold/=2

        self.MLE_cat_y_cap =  np.where(self.MLE_ori_sampled_y_cap<=(self.MLE_ori_sampled_y_cap.min()+pred_threshold), 0, self.MLE_ori_sampled_y_cap)
        self.MLE_cat_y_cap =  np.where(self.MLE_cat_y_cap>=(self.MLE_ori_sampled_y_cap.max()-pred_threshold), 2, self.MLE_cat_y_cap)
        self.MLE_cat_y_cap =  np.where((0<self.MLE_cat_y_cap) & (self.MLE_cat_y_cap<2), 1, self.MLE_cat_y_cap)

        self.MLE_cat_y_cap = self.MLE_cat_y_cap.astype(int)

        if factual:
            how_many_more = 1-np.count_nonzero(self.MLE_cat_y_cap==2)
        else:
            how_many_more = 1-np.count_nonzero(self.MLE_cat_y_cap==0) 

        counts = np.array([np.count_nonzero(self.MLE_cat_y_cap==val) for val in range(3)])

        # print('how many more:')
        # print(how_many_more)

        if how_many_more>0:

            step = 0

            to_substitute = np.argmax(counts)
            p_perturb = self.adjustPperturb(np.argmin(counts))
            # print(f"new p perturb: {p_perturb}")

            while how_many_more>0 and step<2*num_samples:
                # print(how_many_more)

                step+=1

                if factual:
                    X_perturb = {k:torch.zeros_like(self.computationalG_x_dict[k]) for k in self.computationalG_x_dict}
                else:
                    X_perturb = copy.deepcopy(self.computationalG_x_dict)
                # print('X_perturb')
                # print(X_perturb['author'].device)

                # print('perturbing nodes in S...')
                for n in S:

                    # print('perturbing nodes in S: '+str(n))
                    node = mapping[node_type][n]
                    # print(node)

                    pos = self.computationalG_x_dict[node_type][node].nonzero().cpu().numpy()[raw_feature_exp[n]]
                    # print(pos)

                    perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)

                    tmp = (self.computationalG_x_dict[node_type][node][pos.T]!=perturb_array).cpu().detach().numpy()
                    tmp = np.expand_dims(tmp, axis=0)

                    X_perturb[node_type][node][pos.T] = perturb_array

                    if n not in self.MLE_sampled_data:
                        self.MLE_sampled_data[n] = tmp
                    else:
                        self.MLE_sampled_data[n] = np.append(self.MLE_sampled_data[n], tmp, axis=0)

                if X_perturb['movie'].device != self.device:        
                    for nt in X_perturb:
                        X_perturb[nt] = X_perturb[nt].to(self.device)
                
                # self.computationalG2device()
                
                with torch.no_grad():
                    out = nn.functional.softmax(self.model(X_perturb, self.computationalG_edge_index_dict), dim=-1)

                pred_score= out[mapping['movie'][target]][self.orig_pred_label].item()  

                if factual and pred_score>=self.MLE_ori_sampled_y_cap.max()-pred_threshold:
                    # print(f"Got one from {2}!")
                    how_many_more-=1
                    # print(np.sum(how_many_more))
                    # print(how_many_more)

                    sub_pos = np.random.choice(np.nonzero(self.MLE_cat_y_cap==1)[0], size=1)

                    self.MLE_cat_y_cap[sub_pos] = 2

                    for key in self.MLE_sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.MLE_sampled_data[key][sub_pos,:]=self.MLE_sampled_data[key][-1:,:]
                        self.MLE_sampled_data[key] = self.MLE_sampled_data[key][:-1, :]   

                elif not factual and pred_score<=self.MLE_ori_sampled_y_cap.min()+pred_threshold:

                    # print(f"Got one from {0}!")
                    how_many_more-=1
                    # print(np.sum(how_many_more))
                    # print(how_many_more)

                    sub_pos = np.random.choice(np.nonzero(self.MLE_cat_y_cap==1)[0], size=1)

                    self.MLE_cat_y_cap[sub_pos] = 0

                    for key in self.MLE_sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.MLE_sampled_data[key][sub_pos,:]=self.MLE_sampled_data[key][-1:,:]
                        self.MLE_sampled_data[key] = self.MLE_sampled_data[key][:-1, :]   
                else:
                    for key in self.MLE_sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.MLE_sampled_data[key] = self.MLE_sampled_data[key][:-1, :]
            if how_many_more>0 and step==10*num_samples:
                self.MLE_sampling_fail = True
                # print('WARNING: MLE sampling failed!')

        return g, mapping


    @torch.no_grad()
    def MLEsamplingMUTAG(self, target, S=[], raw_feature_exp={}, n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, pred_threshold=0.01, factual=True):
        # for u, do element-wise feature sampling; for S, do categorical sampling based on feature explanation
        
        data = self.MUTAG_dataset[target].to(self.device)

        self.MLE_sampling_fail = False

        num_samples = max(k*(data.x.count_nonzero().item()+n_cat_value), num_samples)

        # print("Generating "+str(num_samples)+" samples on target: " + str(target))

        self.MLE_sampled_data = {}

        self.orig_pred, self.orig_pred_label = self.getOrigPred(target)

        sampled_y_cap = []


        for iteration in range(num_samples):
            # print('*** Iter: '+str(iteration))

            if factual:
                X_perturb = torch.zeros_like(data.x)
            else:
                X_perturb = copy.deepcopy(data.x)

            for n in S:
                # print('perturbing u')

                pos = data.x[n].nonzero().cpu().numpy()[raw_feature_exp[n]]
                # print(pos.shape)
                # print(pos)

                perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)

                tmp = (1-perturb_array).cpu().detach().numpy()
                tmp = np.expand_dims(tmp, axis=0)

                if factual:
                    X_perturb[n][pos.T] = data.x[n][pos.T]*perturb_array

                else:
                    X_perturb[n][pos.T] *= perturb_array

                if n not in self.MLE_sampled_data:
                    self.MLE_sampled_data[n] = tmp
                else:
                    self.MLE_sampled_data[n] = np.append(self.MLE_sampled_data[n], tmp, axis=0)

            if X_perturb.device != self.device:
                X_perturb = X_perturb.to(self.device)
            
            with torch.no_grad():
                out = nn.functional.softmax(self.model(X_perturb, data.edge_index, torch.tensor([0 for _ in range(X_perturb.shape[0])]).to(self.device))[0], dim=-1)

            pred_score= out[self.orig_pred_label].item()
            sampled_y_cap.append(pred_score)

        self.ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T

        self.MLE_ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T

        perturb_range = self.MLE_ori_sampled_y_cap.max() - self.MLE_ori_sampled_y_cap.min()

        if perturb_range==0: 

            # print('GNN prediction never change!')
            self.MLE_sampling_fail = True
            self.MLE_cat_y_cap = np.ones(self.MLE_ori_sampled_y_cap.shape)
            return 

        elif perturb_range<pred_threshold:
            # print('perturb range too small, decrease pred_threshold')
            pred_threshold/=2

        self.MLE_cat_y_cap =  np.where(self.MLE_ori_sampled_y_cap<=(self.MLE_ori_sampled_y_cap.min()+pred_threshold), 0, self.MLE_ori_sampled_y_cap)
        self.MLE_cat_y_cap =  np.where(self.MLE_cat_y_cap>=(self.MLE_ori_sampled_y_cap.max()-pred_threshold), 2, self.MLE_cat_y_cap)
        self.MLE_cat_y_cap =  np.where((0<self.MLE_cat_y_cap) & (self.MLE_cat_y_cap<2), 1, self.MLE_cat_y_cap)

        self.MLE_cat_y_cap = self.MLE_cat_y_cap.astype(int)

        if factual:
            how_many_more = 1-np.count_nonzero(self.MLE_cat_y_cap==2)
        else:
            how_many_more = 1-np.count_nonzero(self.MLE_cat_y_cap==0) 

        counts = np.array([np.count_nonzero(self.MLE_cat_y_cap==val) for val in range(3)])

        # print('how many more:')
        # print(how_many_more)

        if how_many_more>0:

            step = 0

            to_substitute = np.argmax(counts)
            p_perturb = self.adjustPperturb(np.argmin(counts))
            # print(f"new p perturb: {p_perturb}")

            while how_many_more>0 and step<2*num_samples:
                # print(how_many_more)

                step+=1

                if factual:
                    X_perturb = torch.zeros_like(data.x)
                else:
                    X_perturb = copy.deepcopy(data.x)

                for n in S:
                    # print('perturbing u')

                    pos = data.x[n].nonzero().cpu().numpy()[raw_feature_exp[n]]
                    # print(pos.shape)
                    # print(pos)

                    perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)

                    tmp = (1-perturb_array).cpu().detach().numpy()
                    tmp = np.expand_dims(tmp, axis=0)

                    if factual:
                        X_perturb[n][pos.T] = data.x[n][pos.T]*perturb_array

                    else:
                        X_perturb[n][pos.T] *= perturb_array

                    if n not in self.MLE_sampled_data:
                        self.MLE_sampled_data[n] = tmp
                    else:
                        self.MLE_sampled_data[n] = np.append(self.MLE_sampled_data[n], tmp, axis=0)

                if X_perturb.device != self.device:
                    X_perturb = X_perturb.to(self.device)
                
                with torch.no_grad():
                    out = nn.functional.softmax(self.model(X_perturb, data.edge_index, torch.tensor([0 for _ in range(X_perturb.shape[0])]).to(self.device))[0], dim=-1)

                pred_score= out[self.orig_pred_label].item()

                if factual and pred_score>=self.MLE_ori_sampled_y_cap.max()-pred_threshold:
                    # print(f"Got one from {2}!")
                    how_many_more-=1
                    # print(np.sum(how_many_more))
                    # print(how_many_more)

                    sub_pos = np.random.choice(np.nonzero(self.MLE_cat_y_cap==1)[0], size=1)

                    self.MLE_cat_y_cap[sub_pos] = 2

                    for key in self.MLE_sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.MLE_sampled_data[key][sub_pos,:]=self.MLE_sampled_data[key][-1:,:]
                        self.MLE_sampled_data[key] = self.MLE_sampled_data[key][:-1, :]   

                elif not factual and pred_score<=self.MLE_ori_sampled_y_cap.min()+pred_threshold:

                    # print(f"Got one from {0}!")
                    how_many_more-=1
                    # print(np.sum(how_many_more))
                    # print(how_many_more)

                    sub_pos = np.random.choice(np.nonzero(self.MLE_cat_y_cap==1)[0], size=1)

                    self.MLE_cat_y_cap[sub_pos] = 0

                    for key in self.MLE_sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.MLE_sampled_data[key][sub_pos,:]=self.MLE_sampled_data[key][-1:,:]
                        self.MLE_sampled_data[key] = self.MLE_sampled_data[key][:-1, :]   
                else:
                    for key in self.MLE_sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.MLE_sampled_data[key] = self.MLE_sampled_data[key][:-1, :]
            if how_many_more>0 and step==10*num_samples:
                self.MLE_sampling_fail = True
                # print('WARNING: MLE sampling failed!')

        return 

    @torch.no_grad()
    def uniformPerturbIMDB(self, target, n_cat_value=3, k=10, p_perturb=0.5, num_samples=1000, pred_threshold=0.01):
        # for u, do element-wise feature sampling; for S, do categorical sampling based on feature explanation
        
        mapping, reversed_mapping, g = self.__subgraph__(target)

        num_RV = n_cat_value

        node_type = 'movie'

        for n in g.nodes:
            # print(n)
            node = mapping[node_type][n]
            num_RV += self.computationalG_x_dict[node_type][node].count_nonzero().item()
            # print(num_RV)

        num_samples = max(k*num_RV, num_samples)
        # print(f'num_samples_needed is {num_samples_needed}')

        print("Generating "+str(num_samples)+" samples on target: " + str(target))

        self.sampled_data = {}

        # self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

        sampled_y_cap = []

        for iteration in range(num_samples):
            # print('*** Iter: '+str(iteration))

            X_perturb = copy.deepcopy(self.computationalG_x_dict)
            # print('X_perturb')
            # print(X_perturb['author'].device)

            for n in g.nodes:
                # print('perturbing u')

                node = mapping[node_type][n]
                
                seed = random.choices([0,1], weights = [1-p_perturb, p_perturb], k=1)[0]
                # seed=1
                pos = self.computationalG_x_dict[node_type][node].nonzero().cpu().numpy()
                # print(pos.shape)

                if seed==1:

                    perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)

                    tmp = (X_perturb[node_type][node][pos.T]!=perturb_array).cpu().detach().numpy()
                    tmp = np.expand_dims(tmp, axis=0)

                    X_perturb[node_type][node][pos.T] = perturb_array

                else:
                    tmp = np.zeros((1, pos.shape[0]), dtype=np.int8)

                if n not in self.sampled_data:
                    self.sampled_data[n] = tmp
                else:
                    self.sampled_data[n] = np.append(self.sampled_data[n], tmp, axis=0)

            if X_perturb['movie'].device != self.device:        
                for nt in X_perturb:
                    X_perturb[nt] = X_perturb[nt].to(self.device)
                        
            with torch.no_grad():
                out = nn.functional.softmax(self.model(X_perturb, self.computationalG_edge_index_dict), dim=-1)

            pred_score= out[mapping['movie'][target]][self.orig_pred_label].item()      
            sampled_y_cap.append(pred_score)

        self.ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T
        perturb_range = self.ori_sampled_y_cap.max() - self.ori_sampled_y_cap.min()

        if perturb_range==0: 
            print('GNN prediction never change!')
            self.cat_y_cap = np.ones(self.ori_sampled_y_cap.shape)

            return g, mapping

        elif perturb_range<pred_threshold:
            print('perturb range too small, decrease pred_threshold')
            pred_threshold/=2

        self.cat_y_cap =  np.where(self.ori_sampled_y_cap<=(self.ori_sampled_y_cap.min()+pred_threshold), 0, self.ori_sampled_y_cap)
        self.cat_y_cap =  np.where(self.cat_y_cap>=(self.ori_sampled_y_cap.max()-pred_threshold), 2, self.cat_y_cap)
        self.cat_y_cap =  np.where((0<self.cat_y_cap) & (self.cat_y_cap<2), 1, self.cat_y_cap)

        self.cat_y_cap = self.cat_y_cap.astype(int)

        # print(f"max percentage: {np.count_nonzero(self.cat_y_cap == 2) / self.cat_y_cap.shape[0]}")
        # print(f"min percentage: {np.count_nonzero(self.cat_y_cap == 0) / self.cat_y_cap.shape[0]}")

        # _, counts = np.unique(self.cat_y_cap, return_counts=True)
        counts = np.array([np.count_nonzero(self.cat_y_cap==val) for val in range(3)])

        bar = 0.001
        how_many_more = np.where(bar*num_samples-counts<0, 0, np.ceil(bar*num_samples-counts)).astype(int)

        # print('how many more:')
        # print(how_many_more)

        if np.sum(how_many_more)>0:
            to_substitute = np.argmax(counts)

            step = 0

            if to_substitute==0 or to_substitute==2:
                p_perturb = self.adjustPperturb(np.argmin(counts))
                print(f"new p perturb: {p_perturb}")

            while np.sum(how_many_more)>0 and step<2*num_samples:
                # print(how_many_more)

                step+=1

                X_perturb = copy.deepcopy(self.computationalG_x_dict)

                for n in g.nodes:
                    # print('perturbing u')
                    node = mapping[node_type][n]

                    seed = random.choices([0, 1], weights=[1 - p_perturb, p_perturb], k=1)[0]

                    pos = self.computationalG_x_dict[node_type][node].nonzero().cpu().numpy()
                    # print(pos.shape)

                    if seed == 1:

                        perturb_array = torch.tensor(np.random.choice(2, size=pos.shape[0], p=[0.5, 0.5]), dtype=torch.float32).to(self.device)

                        tmp = (X_perturb[node_type][node][pos.T] != perturb_array).cpu().detach().numpy()
                        tmp = np.expand_dims(tmp, axis=0)

                        X_perturb[node_type][node][pos.T] = perturb_array

                    else:
                        tmp = np.zeros((1, pos.shape[0]), dtype=np.int8)


                    if n not in self.sampled_data:
                        self.sampled_data[n] = tmp
                    else:
                        self.sampled_data[n] = np.append(self.sampled_data[n], tmp, axis=0)

                if X_perturb['movie'].device != self.device:
                    for nt in X_perturb:
                        X_perturb[nt] = X_perturb[nt].to(self.device)

                with torch.no_grad():
                    out = nn.functional.softmax(self.model(X_perturb, self.computationalG_edge_index_dict), dim=-1)

                pred_score = out[mapping['movie'][target]][self.orig_pred_label].item()

                if how_many_more[0]>0 and pred_score<=self.ori_sampled_y_cap.min()+pred_threshold:
                    print(f"Got one from {0}!")
                    how_many_more[0]-=1
                    # print(np.sum(how_many_more))
                    print(how_many_more)

                    sub_pos = np.random.choice(np.nonzero(self.cat_y_cap==to_substitute)[0], size=1)

                    self.cat_y_cap[sub_pos] = 0

                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key][sub_pos,:]=self.sampled_data[key][-1:,:]
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]

                elif how_many_more[1]>0 and self.ori_sampled_y_cap.min()+pred_threshold<pred_score<self.ori_sampled_y_cap.max()-pred_threshold:

                    print(f"Got one from {1}!")
                    how_many_more[1]-=1
                    print(how_many_more)
                    sub_pos = np.random.choice(np.nonzero(self.cat_y_cap==to_substitute)[0], size=1)

                    self.cat_y_cap[sub_pos] = 1

                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key][sub_pos,:]=self.sampled_data[key][-1:,:]
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]

                elif how_many_more[2]>0 and pred_threshold>=pred_score<self.ori_sampled_y_cap.max()-pred_threshold:

                    print(f"Got one from {2}!")
                    how_many_more[2]-=1
                    print(how_many_more)
                    sub_pos = np.random.choice(np.nonzero(self.cat_y_cap==to_substitute)[0], size=1)

                    self.cat_y_cap[sub_pos] = 2

                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key][sub_pos,:]=self.sampled_data[key][-1:,:]
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]

                else:
                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]
                # print(np.sum(how_many_more))
                # print(step)


        return g, mapping

    @torch.no_grad()
    def uniformPerturbMUTAG(self, target, n_cat_value=3, k=10, p_perturb=0.5, num_samples=1000, pred_threshold=0.01):
        # for u, do element-wise feature sampling; for S, do categorical sampling based on feature explanation
        
        data = self.MUTAG_dataset[target].to(self.device)

        num_samples = max(k*(data.x.count_nonzero().item()+n_cat_value), num_samples)
        # print(f'num_samples_needed is {num_samples_needed}')

        print("Generating "+str(num_samples)+" samples on target: " + str(target))

        self.sampled_data = {}

        # self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

        sampled_y_cap = []

        for iteration in range(num_samples):
            # print('*** Iter: '+str(iteration))

            X_perturb = copy.deepcopy(data.x)
            # .to(self.device)
            # print('X_perturb')
            # print(X_perturb.device)

            for n in range(data.x.shape[0]):
                # print('perturbing u')
                
                seed = random.choices([0,1], weights = [1-p_perturb, p_perturb], k=1)[0]

                pos = data.x[n].nonzero().cpu().numpy()
                # print(pos.shape)

                if seed==1:

                    perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)

                    X_perturb[n][pos.T] *= perturb_array

                    tmp = (1-perturb_array).cpu().detach().numpy()
                    tmp = np.expand_dims(tmp, axis=0)


                else:
                    tmp = np.zeros((1, pos.shape[0]), dtype=np.int8)

                if n not in self.sampled_data:
                    self.sampled_data[n] = tmp
                else:
                    self.sampled_data[n] = np.append(self.sampled_data[n], tmp, axis=0)

            if X_perturb.device != self.device:
                X_perturb = X_perturb.to(self.device)

            with torch.no_grad():
                out = nn.functional.softmax(self.model(X_perturb, data.edge_index, torch.tensor([0 for _ in range(X_perturb.shape[0])]).to(self.device))[0], dim=-1)

            pred_score= out[self.orig_pred_label].item()
            sampled_y_cap.append(pred_score)

        self.ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T
        perturb_range = self.ori_sampled_y_cap.max() - self.ori_sampled_y_cap.min()

        if perturb_range==0: 
            print('GNN prediction never change!')
            self.cat_y_cap = np.ones(self.ori_sampled_y_cap.shape)

            return

        elif perturb_range<pred_threshold:
            print('perturb range too small, decrease pred_threshold')
            pred_threshold/=2

        self.cat_y_cap =  np.where(self.ori_sampled_y_cap<=(self.ori_sampled_y_cap.min()+pred_threshold), 0, self.ori_sampled_y_cap)
        self.cat_y_cap =  np.where(self.cat_y_cap>=(self.ori_sampled_y_cap.max()-pred_threshold), 2, self.cat_y_cap)
        self.cat_y_cap =  np.where((0<self.cat_y_cap) & (self.cat_y_cap<2), 1, self.cat_y_cap)

        self.cat_y_cap = self.cat_y_cap.astype(int)

        # print(f"max percentage: {np.count_nonzero(self.cat_y_cap == 2) / self.cat_y_cap.shape[0]}")
        # print(f"min percentage: {np.count_nonzero(self.cat_y_cap == 0) / self.cat_y_cap.shape[0]}")

        # _, counts = np.unique(self.cat_y_cap, return_counts=True)
        counts = np.array([np.count_nonzero(self.cat_y_cap==val) for val in range(3)])

        bar = 0.001
        how_many_more = np.where(bar*num_samples-counts<0, 0, np.ceil(bar*num_samples-counts)).astype(int)

        # print('how many more:')
        # print(how_many_more)

        if np.sum(how_many_more)>0:
            to_substitute = np.argmax(counts)

            step = 0

            if to_substitute==0 or to_substitute==2:
                p_perturb = self.adjustPperturb(np.argmin(counts))
                print(f"new p perturb: {p_perturb}")

            while np.sum(how_many_more)>0 and step<2*num_samples:
                # print(how_many_more)

                step+=1

                X_perturb = copy.deepcopy(data.x)
                # print('X_perturb')
                # print(X_perturb['author'].device)

                for n in range(data.x.shape[0]):
                    # print('perturbing u')

                    seed = random.choices([0, 1], weights=[1 - p_perturb, p_perturb], k=1)[0]

                    pos = data.x[n].nonzero().cpu().numpy()
                    # print(pos.shape)

                    if seed == 1:

                        perturb_array = torch.tensor(
                            np.random.choice(2, size=pos.shape[0], p=[1 - p_perturb, p_perturb]),
                            dtype=torch.float32).to(self.device)

                        tmp = (1 - perturb_array).cpu().detach().numpy()
                        tmp = np.expand_dims(tmp, axis=0)

                        X_perturb[n][pos.T] *= perturb_array

                    else:
                        tmp = np.zeros((1, pos.shape[0]), dtype=np.int8)

                    if n not in self.sampled_data:
                        self.sampled_data[n] = tmp
                    else:
                        self.sampled_data[n] = np.append(self.sampled_data[n], tmp, axis=0)

                if X_perturb.device != self.device:
                    X_perturb = X_perturb.to(self.device)

                with torch.no_grad():
                    out = nn.functional.softmax(
                        self.model(X_perturb, data.edge_index, torch.tensor([0 for _ in range(X_perturb.shape[0])]).to(self.device))[0],
                        dim=-1)

                pred_score = out[self.orig_pred_label].item()

                if how_many_more[0]>0 and pred_score<=self.ori_sampled_y_cap.min()+pred_threshold:
                    print(f"Got one from {0}!")
                    how_many_more[0]-=1
                    # print(np.sum(how_many_more))
                    print(how_many_more)

                    sub_pos = np.random.choice(np.nonzero(self.cat_y_cap==to_substitute)[0], size=1)

                    self.cat_y_cap[sub_pos] = 0

                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key][sub_pos,:]=self.sampled_data[key][-1:,:]
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]

                elif how_many_more[1]>0 and self.ori_sampled_y_cap.min()+pred_threshold<pred_score<self.ori_sampled_y_cap.max()-pred_threshold:

                    print(f"Got one from {1}!")
                    how_many_more[1]-=1
                    print(how_many_more)
                    sub_pos = np.random.choice(np.nonzero(self.cat_y_cap==to_substitute)[0], size=1)

                    self.cat_y_cap[sub_pos] = 1

                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key][sub_pos,:]=self.sampled_data[key][-1:,:]
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]

                elif how_many_more[2]>0 and pred_threshold>=pred_score<self.ori_sampled_y_cap.max()-pred_threshold:

                    print(f"Got one from {2}!")
                    how_many_more[2]-=1
                    print(how_many_more)
                    sub_pos = np.random.choice(np.nonzero(self.cat_y_cap==to_substitute)[0], size=1)

                    self.cat_y_cap[sub_pos] = 2

                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key][sub_pos,:]=self.sampled_data[key][-1:,:]
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]

                else:
                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]
                # print(np.sum(how_many_more))
                # print(step)
        return

    # @torch.no_grad()
    # def uniformPerturbDBLP(self, target, n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, pred_threshold=0.01):
    #     # element-wise feature perturbing for all nodes in g

    #     mapping, reversed_mapping, g = self.__subgraph__(target)

    #     num_RV = n_cat_value

    #     for n in g.nodes:
    #         # print(n)
    #         node_type = self.DBLP_idx2node_type[n%10]
    #         node = mapping[node_type][n//10]
    #         num_RV += self.computationalG_x_dict[node_type][node].count_nonzero().item()
    #         # print(num_RV)
            
    #     num_samples = max(k*num_RV, num_samples)

    #     print("Generating "+str(num_samples)+" samples on target: " + str(target))

    #     self.sampled_data = {}
        
    #     # self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

    #     sampled_y_cap = []

    #     for iteration in range(num_samples):
    #         # print('*** Iter: '+str(iteration))

    #         X_perturb = copy.deepcopy(self.computationalG_x_dict)
    #         # print('X_perturb')
    #         # print(X_perturb['author'].device)

    #         for n in g.nodes:
    #             # print('perturbing u')
    #             node_type = self.DBLP_idx2node_type[n%10]
    #             node = mapping[node_type][n//10]
                
    #             seed = random.choices([0,1], weights = [1-p_perturb, p_perturb], k=1)[0]

    #             if node_type=='author' or node_type=='paper':

    #                 pos = self.computationalG_x_dict[node_type][node].nonzero().cpu().numpy()
    #                 # print(pos.shape)

    #                 if seed==1:

    #                     # perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)
    #                     perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[0.5, 0.5]), dtype=torch.float32).to(self.device)

    #                     tmp = (X_perturb[node_type][node][pos.T]!=perturb_array).cpu().detach().numpy()
    #                     tmp = np.expand_dims(tmp, axis=0)

    #                     X_perturb[node_type][node][pos.T] = perturb_array

    #                 else:
    #                     tmp = np.zeros((1, pos.shape[0]), dtype=np.int8)

    #             elif node_type=='conference':
    #                 # print(node)

    #                 if seed==1:

    #                     X_perturb[node_type][node] = 0 

    #                     tmp = np.ones((1,1), dtype=np.int8)

    #                 else:
    #                     tmp = np.zeros((1,1), dtype=np.int8)

    #             if n not in self.sampled_data:
    #                 self.sampled_data[n] = tmp
    #             else:
    #                 self.sampled_data[n] = np.append(self.sampled_data[n], tmp, axis=0)

    #         if X_perturb['author'].device != self.device:        
    #             for nt in X_perturb:
    #                 X_perturb[nt] = X_perturb[nt].to(self.device)

    #         with torch.no_grad():
    #             out = nn.functional.softmax(self.model(X_perturb, self.computationalG_edge_index_dict), dim=-1)

    #         pred_score= out[mapping['author'][target]][self.orig_pred_label].item()      
    #         sampled_y_cap.append(pred_score)


    #     self.ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T
    #     perturb_range = self.ori_sampled_y_cap.max() - self.ori_sampled_y_cap.min()

    #     if perturb_range==0: 
    #         print('GNN prediction never change!')
    #         self.cat_y_cap = np.ones(self.ori_sampled_y_cap.shape)

    #         return g, mapping

    #     elif perturb_range<pred_threshold:
    #         print('perturb range too small, decrease pred_threshold')
    #         pred_threshold/=2

    #     self.cat_y_cap =  np.where(self.ori_sampled_y_cap<=(self.ori_sampled_y_cap.min()+pred_threshold), 0, self.ori_sampled_y_cap)
    #     self.cat_y_cap =  np.where(self.cat_y_cap>=(self.ori_sampled_y_cap.max()-pred_threshold), 2, self.cat_y_cap)
    #     self.cat_y_cap =  np.where((0<self.cat_y_cap) & (self.cat_y_cap<2), 1, self.cat_y_cap)

    #     self.cat_y_cap = self.cat_y_cap.astype(int)

    #     return g, mapping

    @torch.no_grad()
    def uniformPerturbDBLP(self, target, n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, pred_threshold=0.01):
        # element-wise feature perturbing for all nodes in g

        mapping, reversed_mapping, g = self.__subgraph__(target)

        num_RV = n_cat_value

        for n in g.nodes:
            # print(n)
            node_type = self.DBLP_idx2node_type[n%10]
            node = mapping[node_type][n//10]
            num_RV += self.computationalG_x_dict[node_type][node].count_nonzero().item()
            # print(num_RV)
            
        num_samples = max(k*num_RV, num_samples)

        print("Generating "+str(num_samples)+" samples on target: " + str(target))

        self.sampled_data = {}
        
        # self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

        sampled_y_cap = []

        for iteration in range(num_samples):
            # print('*** Iter: '+str(iteration))

            X_perturb = copy.deepcopy(self.computationalG_x_dict)
            # print('X_perturb')
            # print(X_perturb['author'].device)

            for n in g.nodes:
                # print('perturbing u')
                node_type = self.DBLP_idx2node_type[n%10]
                node = mapping[node_type][n//10]
                
                seed = random.choices([0,1], weights = [1-p_perturb, p_perturb], k=1)[0]

                if node_type=='author' or node_type=='paper':

                    pos = self.computationalG_x_dict[node_type][node].nonzero().cpu().numpy()
                    # print(pos.shape)

                    if seed==1:

                        # perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)
                        perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[0.5, 0.5]), dtype=torch.float32).to(self.device)

                        tmp = (X_perturb[node_type][node][pos.T]!=perturb_array).cpu().detach().numpy()
                        tmp = np.expand_dims(tmp, axis=0)

                        X_perturb[node_type][node][pos.T] = perturb_array

                    else:
                        tmp = np.zeros((1, pos.shape[0]), dtype=np.int8)

                elif node_type=='conference':
                    # print(node)

                    if seed==1:

                        X_perturb[node_type][node] = 0 

                        tmp = np.ones((1,1), dtype=np.int8)

                    else:
                        tmp = np.zeros((1,1), dtype=np.int8)

                if n not in self.sampled_data:
                    self.sampled_data[n] = tmp
                else:
                    self.sampled_data[n] = np.append(self.sampled_data[n], tmp, axis=0)

            if X_perturb['author'].device != self.device:        
                for nt in X_perturb:
                    X_perturb[nt] = X_perturb[nt].to(self.device)

            with torch.no_grad():
                out = nn.functional.softmax(self.model(X_perturb, self.computationalG_edge_index_dict), dim=-1)

            pred_score= out[mapping['author'][target]][self.orig_pred_label].item()      
            sampled_y_cap.append(pred_score)

        self.ori_sampled_y_cap = np.expand_dims(sampled_y_cap, axis=0).T
        perturb_range = self.ori_sampled_y_cap.max() - self.ori_sampled_y_cap.min()

        if perturb_range==0: 
            print('GNN prediction never change!')
            self.cat_y_cap = np.ones(self.ori_sampled_y_cap.shape)

            return g, mapping

        elif perturb_range<pred_threshold:
            print('perturb range too small, decrease pred_threshold')
            pred_threshold/=2

        self.cat_y_cap =  np.where(self.ori_sampled_y_cap<=(self.ori_sampled_y_cap.min()+pred_threshold), 0, self.ori_sampled_y_cap)
        self.cat_y_cap =  np.where(self.cat_y_cap>=(self.ori_sampled_y_cap.max()-pred_threshold), 2, self.cat_y_cap)
        self.cat_y_cap =  np.where((0<self.cat_y_cap) & (self.cat_y_cap<2), 1, self.cat_y_cap)

        self.cat_y_cap = self.cat_y_cap.astype(int)

        # print(f"max percentage: {np.count_nonzero(self.cat_y_cap == 2) / self.cat_y_cap.shape[0]}")
        # print(f"min percentage: {np.count_nonzero(self.cat_y_cap == 0) / self.cat_y_cap.shape[0]}")

        # _, counts = np.unique(self.cat_y_cap, return_counts=True)
        counts = np.array([np.count_nonzero(self.cat_y_cap==val) for val in range(3)])

        bar = 0.001
        how_many_more = np.where(bar*num_samples-counts<0, 0, np.ceil(bar*num_samples-counts)).astype(int)

        # print('how many more:')
        # print(how_many_more)

        if np.sum(how_many_more)>0:
            to_substitute = np.argmax(counts)

            step = 0

            if to_substitute==0 or to_substitute==2:
                p_perturb = self.adjustPperturb(np.argmin(counts))
                print(f"new p perturb: {p_perturb}")

            while np.sum(how_many_more)>0 and step<2*num_samples:
                # print(how_many_more)

                step+=1

                X_perturb = copy.deepcopy(self.computationalG_x_dict)

                for n in g.nodes:
                    # print('perturbing u')
                    node_type = self.DBLP_idx2node_type[n % 10]
                    node = mapping[node_type][n // 10]

                    seed = random.choices([0, 1], weights=[1 - p_perturb, p_perturb], k=1)[0]

                    if node_type == 'author' or node_type == 'paper':

                        pos = self.computationalG_x_dict[node_type][node].nonzero().cpu().numpy()
                        # print(pos.shape)

                        if seed == 1:

                            # perturb_array =  torch.tensor(np.random.choice(2, size=pos.shape[0], p =[1-p_perturb, p_perturb]), dtype=torch.float32).to(self.device)
                            perturb_array = torch.tensor(np.random.choice(2, size=pos.shape[0], p=[0.5, 0.5]), dtype=torch.float32).to(self.device)

                            tmp = (X_perturb[node_type][node][pos.T] != perturb_array).cpu().detach().numpy()
                            tmp = np.expand_dims(tmp, axis=0)

                            X_perturb[node_type][node][pos.T] = perturb_array

                        else:
                            tmp = np.zeros((1, pos.shape[0]), dtype=np.int8)

                    elif node_type == 'conference':
                        # print(node)

                        if seed == 1:

                            X_perturb[node_type][node] = 0

                            tmp = np.ones((1, 1), dtype=np.int8)

                        else:
                            tmp = np.zeros((1, 1), dtype=np.int8)

                    if n not in self.sampled_data:
                        self.sampled_data[n] = tmp
                    else:
                        self.sampled_data[n] = np.append(self.sampled_data[n], tmp, axis=0)

                if X_perturb['author'].device != self.device:
                    for nt in X_perturb:
                        X_perturb[nt] = X_perturb[nt].to(self.device)

                with torch.no_grad():
                    out = nn.functional.softmax(self.model(X_perturb, self.computationalG_edge_index_dict), dim=-1)

                pred_score = out[mapping['author'][target]][self.orig_pred_label].item()

                if how_many_more[0]>0 and pred_score<=self.ori_sampled_y_cap.min()+pred_threshold:
                    print(f"Got one from {0}!")
                    how_many_more[0]-=1
                    # print(np.sum(how_many_more))
                    print(how_many_more)

                    sub_pos = np.random.choice(np.nonzero(self.cat_y_cap==to_substitute)[0], size=1)

                    self.cat_y_cap[sub_pos] = 0

                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key][sub_pos,:]=self.sampled_data[key][-1:,:]
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]

                elif how_many_more[1]>0 and self.ori_sampled_y_cap.min()+pred_threshold<pred_score<self.ori_sampled_y_cap.max()-pred_threshold:

                    print(f"Got one from {1}!")
                    how_many_more[1]-=1
                    print(how_many_more)
                    sub_pos = np.random.choice(np.nonzero(self.cat_y_cap==to_substitute)[0], size=1)

                    self.cat_y_cap[sub_pos] = 1

                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key][sub_pos,:]=self.sampled_data[key][-1:,:]
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]

                elif how_many_more[2]>0 and pred_threshold>=pred_score<self.ori_sampled_y_cap.max()-pred_threshold:

                    print(f"Got one from {2}!")
                    how_many_more[2]-=1
                    print(how_many_more)
                    sub_pos = np.random.choice(np.nonzero(self.cat_y_cap==to_substitute)[0], size=1)

                    self.cat_y_cap[sub_pos] = 2

                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key][sub_pos,:]=self.sampled_data[key][-1:,:]
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]

                else:
                    for key in self.sampled_data:
                        # print(self.sampled_data[key].shape)
                        self.sampled_data[key] = self.sampled_data[key][:-1, :]
                # print(np.sum(how_many_more))
                # print(step)

        return g, mapping

    def adjustPperturb(self, val):
        p_rate = 0
        for node in self.sampled_data.keys():
            # print(node)
            # print(np.count_nonzero(np.sum(explainer.sampled_data[node][(explainer.cat_y_cap==val).nonzero()[0],:], axis=1))/explainer.sampled_data[node][(explainer.cat_y_cap==val).nonzero()[0],:].shape[0])
            p_rate += np.count_nonzero(
                np.sum(self.sampled_data[node][(self.cat_y_cap!=val).nonzero()[0], :], axis=1))/np.count_nonzero(self.cat_y_cap!=val)
            # print(np.count_nonzero(explainer.sampled_data[node][(explainer.cat_y_cap==0).nonzero()[0],:])/explainer.sampled_data[node][(explainer.cat_y_cap==0).nonzero()[0],:].size)
            # p_rate+=np.count_nonzero(explainer.sampled_data[node][(explainer.cat_y_cap==0).nonzero()[0],:])/explainer.sampled_data[node][(explainer.cat_y_cap==0).nonzero()[0],:].size
        # print(f"Avg. {p_rate / len(self.sampled_data)}")
        return p_rate / len(self.sampled_data)

    # def explainFeatureMLEresampling(self, u, basket, S=[], p_threshold=.05): # u is the node to explain

    #     # if self.dataset=='DBLP' and u%10 not in self.DBLP_idx2node_type:
    #     #     raise ValueError("DBLP illegal explain feature")
        
    #     if u not in self.sampled_data:
    #         raise ValueError("u not in sampled data")

    #     print('Explaning features for node: ' + str(u))

    #     S_data = self.cat_y_cap.astype(np.int8)

    #     for n in S:
    #         cat = self.vec2categ(self.sampled_data[n]) # trick to reduce number of columns 
    #         S_data = np.concatenate((S_data, cat), axis=1)

    #     pdData = np.concatenate((S_data, self.sampled_data[u]), axis=1)
    #     # print(pdData.shape)

    #     pdData = pd.DataFrame(pdData)

    #     ind_ori_to_sub = dict(zip(['target'] + S +['f'+str(i) for i in basket], list(pdData.columns)))
    #     # print(ind_ori_to_sub.keys())

    #     # feat_p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[ind_ori_to_sub[n] for n in S], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in range(self.sampled_data[u].shape[1])]]
    #     feat_p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[ind_ori_to_sub[n] for n in S], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in basket]]

    #     feats_to_pick = [i for i, x in enumerate(feat_p_values) if x < p_threshold]

    #     if len(feats_to_pick)==0:
    #         print('No feature is critical')
    #         return None, None

    #     else:
    #         # print(feats_to_pick)
    #         return np.array(feats_to_pick), basket[np.array(feats_to_pick)]


    def explainFeature(self, target_, u, basket, S=[], p_threshold=.05): # u is the node to explain

        # if self.dataset=='DBLP' and u%10 not in self.DBLP_idx2node_type:
        #     raise ValueError("DBLP illegal explain feature")
        
        if u not in self.sampled_data:
            raise ValueError("u not in sampled data")

        # print('Explaning features for node: ' + str(u))

        S_data = self.cat_y_cap.astype(np.int8)

        for n in S:
            cat = self.vec2categ(self.sampled_data[n]) # trick to reduce number of columns 
            S_data = np.concatenate((S_data, cat), axis=1)

        pdData = np.concatenate((S_data, self.sampled_data[u][:, basket]), axis=1)
        # print(pdData.shape)

        pdData = pd.DataFrame(pdData)

        ind_ori_to_sub = dict(zip(['target'] + S +['f'+str(i) for i in basket], list(pdData.columns)))
        # print(ind_ori_to_sub.keys())

        # feat_p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[ind_ori_to_sub[n] for n in S], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in range(self.sampled_data[u].shape[1])]]
        feat_p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[ind_ori_to_sub[n] for n in S], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in basket]]

        # if len(basket)>20 and u==target_: # basket for target node is too large, must introduce conditional independence test
        #     feats_to_pick = []
        #     for idx in np.argsort(-1*np.asarray(feat_p_values)): # trick to reduce conditional
        #         # print(f"f{basket[idx]}")
        #         if feat_p_values[idx]<p_threshold:
        #             if g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub['f'+str(basket[idx])], Z=[ind_ori_to_sub[n] for n in S]+[ind_ori_to_sub['f'+str(basket[idx_])] for idx_ in feats_to_pick], data=pdData, boolean=False)[1]<p_threshold:
        #                 # print(f"feat {basket[idx]} is added.")
        #                 feats_to_pick.append(idx)
        #                 # print(feats_to_pick)
        # else:
        #     feats_to_pick = [i for i, x in enumerate(feat_p_values) if x < p_threshold]

        feats_to_pick = [i for i, x in enumerate(feat_p_values) if x < p_threshold]

        if len(feats_to_pick)==0:
            # print('No feature is critical')
            return None, None

        else:
            # print(feats_to_pick)
            feats_to_pick.sort()
            return np.array(feats_to_pick), basket[np.array(feats_to_pick)]

    # def explainMLEresampling(self, target,  n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, p_threshold=0.05, pred_threshold=0.01):

    #     begin = time.time()

    #     self.computationalG_x_dict = None
    #     self.computationalG_edge_index_dict = None

    #     mapping, _, g_orig = self.__subgraph__(target) 

    #     # print(self.computationalG_x_dict['author'].device)

    #     self.computationalG2device()
    #     # print(self.computationalG_x_dict['author'].device)

    #     self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

    #     # if self.dataset=='DBLP':
    #     #     _, _ = self.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
    #     # elif self.dataset=='IMDB':
    #     #     _, _ = self.uniformPerturbIMDB(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

    #     blanket_basket = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

    #     if len(blanket_basket)==0:
    #         print('len(blanket_basket)==0')
    #         end = time.time()
    #         return [], {}, {}, end-begin

    #     print('Blanket basket:')
    #     print(blanket_basket)


    #     cand_nodes = list(blanket_basket.keys())
    #     # print('cand_nodes are ')
    #     # print(cand_nodes)

    #     if g_orig.number_of_nodes()-len(cand_nodes)==0:
    #         g = g_orig
    #     else:
    #         g = g_orig.subgraph(cand_nodes)

    #     _, _ = self.MLEsamplingDBLP(target, S=list(blanket_basket.keys()), raw_feature_exp=blanket_basket, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False) # sampling for B^\prime

    #     # print(self.sampled_data.keys())

    #     self.node_var = {node:self.syntheticNodeRV(self.sampled_data[node]) for node in blanket_basket} #after MLE sampling, only perturbed features are recorded in sampled data, so only features in basket are recorded.

    #     # print(self.node_var.keys())
        
    #     # print('Input g')
    #     # print(g.nodes)
    #     print(' +++ cut nodes number: '+str(g_orig.number_of_nodes()-g.number_of_nodes()))

    #     if self.dataset=='DBLP':
    #         target_ = target*10
    #     elif self.dataset=='IMDB':
    #         target_ = target

    #     # S = set()
    #     S = []
    #     U = [target_]
    #     I = set()
    #     raw_feature_exp = {}

    #     p_values = [-1]
    #     c = 0

    #     while len(U)>0 and min(p_values)<p_threshold:

    #         u = U[p_values.index(min(p_values))]
    #         # U = set([U[i] for i in range(len(U)) if p_values[i]<p_threshold])

    #         I = I.union([U[i] for i in range(len(U)) if p_values[i]>=p_threshold])

    #         print(' --- Round '+str(c) + ', picked '+str(u))

    #         U = set(U)
    #         U.remove(u)

    #         feats_to_pick, raw_feature_exp[u] = self.explainFeature(u, blanket_basket[u], S=S, p_threshold=p_threshold) # feats_to_pick is the raw pos of picked feats relative to basket

    #         if raw_feature_exp[u] is None: # synthetic node overestimates feature importance
    #             raw_feature_exp.pop(u)
    #             I.add(u)
    #             # print('Feature exp invalid for node '+str(u))
    #             # break

    #         else:
    #             print(f'Fxplanation on {u}: ')
    #             print(raw_feature_exp[u])
    #             self.sampled_data[u] = self.sampled_data[u][:,feats_to_pick]
    #             S.append(u)
    #             U = U.union(set(g[u]).difference(set(S)))

    #         U = U.difference(I)

    #         print('S, U, I are:')
    #         print(S)
    #         print(U)
    #         print(I)

    #         if len(U)>0: # only when U is not empty, compute p values

    #             U = list(U) # use list to fix order

    #             pdData = self.cat_y_cap.astype(np.int8)

    #             for node in S: # conditioned on the current E
    #                 cat = self.vec2categ(self.sampled_data[node]) # trick to reduce column number
    #                 pdData = np.concatenate((pdData, cat), axis=1)

    #             for node in U:
    #                 # print(node)
    #                 pdData = np.concatenate((pdData, self.node_var[node]), axis=1) # compute dependency of synthetic node variable
    #                 # print(pdData.shape)

    #             pdData = pd.DataFrame(pdData)

    #             ind_ori_to_sub = dict(zip(['target'] + S + U, list(pdData.columns)))
    #             # print(ind_ori_to_sub.keys())

    #             # p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[node_] for node_ in S], data=pdData, boolean=False)[1] for node in U]
    #             p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[node_] for node_ in S], data=pdData, boolean=False)[1] for node in U]
    #             # print(ind_ori_to_sub.keys())
    #             # print(p_values[-1])

    #             print('P values: ')
    #             print(p_values)
    #             c+=1

    #     if len(S)==0:
    #         print('len(S)==0')
    #         end = time.time()

    #         return list(S), {}, {}, end-begin

    #     if len(S)>1:
    #         S = self.blanketShrinking(g_orig, list(S), p_threshold)
    #         raw_feature_exp = {k: v for k, v in raw_feature_exp.items() if k in S}

    #     end = time.time()

    #     # sparsity = {}
    #     feature_exp = {}

    #     if self.dataset=='DBLP':
    #         for node in raw_feature_exp:
    #             node_type = self.DBLP_idx2node_type[node%10]
    #             feature_exp[node] = self.computationalG_x_dict[node_type][
    #                 mapping[node_type][node//10]].nonzero().cpu().numpy().T[0][raw_feature_exp[node]]

    #     elif self.dataset=='IMDB':
    #         node_type = 'movie'
    #         for node in raw_feature_exp:
    #             feature_exp[node] = self.computationalG_x_dict[node_type][
    #                 mapping[node_type][node]].nonzero().cpu().numpy().T[0][raw_feature_exp[node]]
    #             # sparsity[node] = raw_feature_exp[node].size/self.computationalG_x_dict[node_type][mapping[node]].count_nonzero().item()

    #     print(' *** final output')
    #     print(S)
    #     print(raw_feature_exp)
    #     print('time used: {:.2f}'.format(end-begin))

    #     return list(S), raw_feature_exp, feature_exp, end-begin

    def explain(self, target,  n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, p_threshold=0.05, pred_threshold=0.01):

        begin = time.time()

        self.computationalG_x_dict = None
        self.computationalG_edge_index_dict = None



        if self.dataset=='DBLP':

            mapping, _, g_orig = self.__subgraph__(target) 
            self.computationalG2device()
            # print(self.computationalG_x_dict['author'].device)

            self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)
            target_ = target*10
            _, _ = self.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            blanket_basket = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

        elif self.dataset=='IMDB':

            mapping, _, g_orig = self.__subgraph__(target) 
            self.computationalG2device()
            # print(self.computationalG_x_dict['author'].device)

            self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

            target_ = target
            _, _ = self.uniformPerturbIMDB(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            blanket_basket = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

        elif self.dataset=='MUTAG':

            g_orig = to_networkx(self.MUTAG_dataset[target], to_undirected=True)

            # self.computationalG2device()
            # print(self.computationalG_x_dict['author'].device)

            self.orig_pred, self.orig_pred_label = self.getOrigPred(target)

            self.uniformPerturbMUTAG(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            blanket_basket, target_ = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
            # blanket_basket = {k:np.array(list(range(self.sampled_data[k].shape[1]))) for k in self.sampled_data}

        if len(blanket_basket)==0 or target_ not in blanket_basket:
            # print('len(blanket_basket)==0 or target not in blanket_basket')
            end = time.time()
            return [], {}, {}, end-begin

        # if len(blanket_basket)==1:
        #     print('len(blanket_basket)==1')
        #     self.sampled_data[target_] = self.sampled_data[target_][:,blanket_basket[target_]]

        #     end = time.time()
        #     return list(blanket_basket.keys()), blanket_basket, self.featureExpRaw2out(blanket_basket, mapping), end-begin

        # print('Blanket basket:')
        # print(blanket_basket)
        
        cand_nodes = list(blanket_basket.keys())
        # print('cand_nodes are ')
        # print(cand_nodes)

        if g_orig.number_of_nodes()-len(cand_nodes)==0:
            g = g_orig
        else:
            g = g_orig.subgraph(cand_nodes)

        # _, _ = self.MLEsamplingDBLP(target, S=list(blanket_basket.keys()), raw_feature_exp=blanket_basket, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False) # sampling for B^\prime

        # print(self.sampled_data.keys())

        # self.node_var = {node: self.syntheticNodeRV(self.sampled_data[node][:, blanket_basket[node]]) for node in blanket_basket} #after MLE sampling, only perturbed features are recorded in sampled data, so only features in basket are recorded.
        self.node_var = {node: self.vec2categ(self.sampled_data[node][:, blanket_basket[node]]) for node in blanket_basket} #after MLE sampling, only perturbed features are recorded in sampled data, so only features in basket are recorded.

        # print(self.node_var.keys())
        
        # print('Input g')
        # print(g.nodes)
        # print(' +++ cut nodes number: '+str(g_orig.number_of_nodes()-g.number_of_nodes()))

        S = []
        U = [target_]
        I = set()
        raw_feature_exp = {}

        p_values = [-1]
        c = 0

        while len(U)>0 and min(p_values)<p_threshold:

            u = U[p_values.index(min(p_values))]
            # U = set([U[i] for i in range(len(U)) if p_values[i]<p_threshold])

            I = I.union([U[i] for i in range(len(U)) if p_values[i]>=p_threshold])

            # print(' --- Round '+str(c) + ', picked '+str(u))

            U = set(U)
            U.remove(u)

            feats_to_pick, raw_feature_exp[u] = self.explainFeature(target_, u, blanket_basket[u], S=S, p_threshold=p_threshold) # feats_to_pick is the raw pos of picked feats relative to basket

            if raw_feature_exp[u] is None: # synthetic node overestimates feature importance
                raw_feature_exp.pop(u)
                I.add(u)
                # print('Feature exp invalid for node '+str(u))
                # break

            else:
                # print(f'Fxplanation on {u}: ')
                # print(raw_feature_exp[u])
                self.sampled_data[u] = self.sampled_data[u][:, blanket_basket[u][feats_to_pick]]
                S.append(u)
                U = U.union(set(g[u]).difference(set(S)))

            U = U.difference(I)

            # print('S, U, I are:')
            # print(S)
            # print(U)
            # print(I)

            if len(U)>0: # only when U is not empty, compute p values

                U = list(U) # use list to fix order

                pdData = self.cat_y_cap.astype(np.int8)

                for node in S: # conditioned on the current E
                    cat = self.vec2categ(self.sampled_data[node]) # trick to reduce column number
                    pdData = np.concatenate((pdData, cat), axis=1)

                for node in U:
                    # print(node)
                    pdData = np.concatenate((pdData, self.node_var[node]), axis=1) # compute dependency of synthetic node variable
                    # print(pdData.shape)

                pdData = pd.DataFrame(pdData)

                ind_ori_to_sub = dict(zip(['target'] + S + U, list(pdData.columns)))
                # print(ind_ori_to_sub.keys())

                # p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[node_] for node_ in S], data=pdData, boolean=False)[1] for node in U]
                p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[node_] for node_ in S], data=pdData, boolean=False)[1] for node in U]
                # print(ind_ori_to_sub.keys())
                # print(p_values[-1])

                # print('P values: ')
                # print(p_values)
                c+=1

        if len(S)==0:
            # print('len(S)==0')
            end = time.time()

            return list(S), {}, {}, end-begin

        if len(S)>1:
            S = self.blanketShrinking(target_, g_orig, list(S), p_threshold)
            raw_feature_exp = {k: v for k, v in raw_feature_exp.items() if k in S}

        end = time.time()

        if self.dataset=='MUTAG':
            feature_exp = self.featureExpRaw2out(raw_feature_exp, target=target)
        else:
            feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping=mapping)

        # sparsity = {}
        # feature_exp = {}
        # if self.dataset=='DBLP':
        #     for node in raw_feature_exp:
        #         node_type = self.DBLP_idx2node_type[node%10]
        #         feature_exp[node] = self.computationalG_x_dict[node_type][
        #             mapping[node_type][node//10]].nonzero().cpu().numpy().T[0][raw_feature_exp[node]]

        # elif self.dataset=='IMDB':
        #     node_type = 'movie'
        #     for node in raw_feature_exp:
        #         feature_exp[node] = self.computationalG_x_dict[node_type][
        #             mapping[node_type][node]].nonzero().cpu().numpy().T[0][raw_feature_exp[node]]
        #         # sparsity[node] = raw_feature_exp[node].size/self.computationalG_x_dict[node_type][mapping[node]].count_nonzero().item()

        # print(' *** final output')
        # print(S)
        # print(raw_feature_exp)
        # print('time used: {:.2f}'.format(end-begin))

        return list(S), raw_feature_exp, feature_exp, end-begin

    def explainNoBasket(self, target,  n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, p_threshold=0.05, pred_threshold=0.01):

        begin = time.time()

        self.computationalG_x_dict = None
        self.computationalG_edge_index_dict = None



        if self.dataset=='DBLP':

            mapping, _, g_orig = self.__subgraph__(target) 
            self.computationalG2device()
            # print(self.computationalG_x_dict['author'].device)

            self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)
            target_ = target*10
            _, _ = self.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            blanket_basket = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

        elif self.dataset=='IMDB':

            mapping, _, g_orig = self.__subgraph__(target) 
            self.computationalG2device()
            # print(self.computationalG_x_dict['author'].device)

            self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

            target_ = target
            _, _ = self.uniformPerturbIMDB(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            blanket_basket = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

        elif self.dataset=='MUTAG':

            g_orig = to_networkx(self.MUTAG_dataset[target], to_undirected=True)

            # self.computationalG2device()
            # print(self.computationalG_x_dict['author'].device)

            self.orig_pred, self.orig_pred_label = self.getOrigPred(target)

            self.uniformPerturbMUTAG(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            blanket_basket, target_ = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
            # blanket_basket = {k:np.array(list(range(self.sampled_data[k].shape[1]))) for k in self.sampled_data}

        pale_blanket_basket = {u: np.array([i for i in range(self.sampled_data[u].shape[1])]) for u in self.sampled_data if self.sampled_data[u].size!=0}
        
        # if len(blanket_basket)==0 or target_ not in blanket_basket:
        #     # print('len(blanket_basket)==0 or target not in blanket_basket')
        #     end = time.time()
        #     return [], {}, {}, end-begin
        
        cand_nodes = list(blanket_basket.keys())
        # print('cand_nodes are ')
        # print(cand_nodes)

        if g_orig.number_of_nodes()-len(cand_nodes)==0:
            g = g_orig
        else:
            g = g_orig.subgraph(cand_nodes)

        self.node_var = {node: self.vec2categ(self.sampled_data[node][:, blanket_basket[node]]) for node in blanket_basket} #after MLE sampling, only perturbed features are recorded in sampled data, so only features in basket are recorded.

        # print(self.node_var.keys())
        
        # print('Input g')
        # print(g.nodes)
        # print(' +++ cut nodes number: '+str(g_orig.number_of_nodes()-g.number_of_nodes()))

        S = []
        U = [target_]
        I = set()
        raw_feature_exp = {}

        p_values = [-1]
        c = 0

        while len(U)>0 and min(p_values)<p_threshold:

            u = U[p_values.index(min(p_values))]
            # U = set([U[i] for i in range(len(U)) if p_values[i]<p_threshold])

            I = I.union([U[i] for i in range(len(U)) if p_values[i]>=p_threshold])

            # print(' --- Round '+str(c) + ', picked '+str(u))

            U = set(U)
            U.remove(u)

            feats_to_pick, raw_feature_exp[u] = self.explainFeature(target_, u, pale_blanket_basket[u], S=S, p_threshold=p_threshold) # feats_to_pick is the raw pos of picked feats relative to basket

            if raw_feature_exp[u] is None: # synthetic node overestimates feature importance
                raw_feature_exp.pop(u)
                I.add(u)
                # print('Feature exp invalid for node '+str(u))
                # break

            else:
                # print(f'Fxplanation on {u}: ')
                # print(raw_feature_exp[u])
                self.sampled_data[u] = self.sampled_data[u][:, pale_blanket_basket[u][feats_to_pick]]
                S.append(u)
                U = U.union(set(g[u]).difference(set(S)))

            U = U.difference(I)

            # print('S, U, I are:')
            # print(S)
            # print(U)
            # print(I)

            if len(U)>0: # only when U is not empty, compute p values

                U = list(U) # use list to fix order

                pdData = self.cat_y_cap.astype(np.int8)

                for node in S: # conditioned on the current E
                    cat = self.vec2categ(self.sampled_data[node]) # trick to reduce column number
                    pdData = np.concatenate((pdData, cat), axis=1)

                for node in U:
                    # print(node)
                    pdData = np.concatenate((pdData, self.node_var[node]), axis=1) # compute dependency of synthetic node variable
                    # print(pdData.shape)

                pdData = pd.DataFrame(pdData)

                ind_ori_to_sub = dict(zip(['target'] + S + U, list(pdData.columns)))
                # print(ind_ori_to_sub.keys())

                # p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[node_] for node_ in S], data=pdData, boolean=False)[1] for node in U]
                p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[node_] for node_ in S], data=pdData, boolean=False)[1] for node in U]
                # print(ind_ori_to_sub.keys())
                # print(p_values[-1])

                # print('P values: ')
                # print(p_values)
                c+=1

        if len(S)==0:
            # print('len(S)==0')
            end = time.time()

            return list(S), {}, {}, end-begin

        if len(S)>1:
            S = self.blanketShrinking(target_, g_orig, list(S), p_threshold)
            raw_feature_exp = {k: v for k, v in raw_feature_exp.items() if k in S}

        end = time.time()

        if self.dataset=='MUTAG':
            feature_exp = self.featureExpRaw2out(raw_feature_exp, target=target)
        else:
            feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping=mapping)


        return list(S), raw_feature_exp, feature_exp, end-begin


    def explainNotConnected(self, target,  n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, p_threshold=0.05, pred_threshold=0.01):

        begin = time.time()

        self.computationalG_x_dict = None
        self.computationalG_edge_index_dict = None



        if self.dataset=='DBLP':

            mapping, _, g_orig = self.__subgraph__(target) 
            self.computationalG2device()
            # print(self.computationalG_x_dict['author'].device)

            self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)
            target_ = target*10
            _, _ = self.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            blanket_basket = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

        elif self.dataset=='IMDB':

            mapping, _, g_orig = self.__subgraph__(target) 
            self.computationalG2device()
            # print(self.computationalG_x_dict['author'].device)

            self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

            target_ = target
            _, _ = self.uniformPerturbIMDB(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            blanket_basket = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)

        elif self.dataset=='MUTAG':

            g_orig = to_networkx(self.MUTAG_dataset[target], to_undirected=True)

            # self.computationalG2device()
            # print(self.computationalG_x_dict['author'].device)

            self.orig_pred, self.orig_pred_label = self.getOrigPred(target)

            self.uniformPerturbMUTAG(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            blanket_basket, target_ = self.basketSearching(target, k=k, p_perturb=p_perturb, p_threshold=p_threshold, pred_threshold=pred_threshold)
            # blanket_basket = {k:np.array(list(range(self.sampled_data[k].shape[1]))) for k in self.sampled_data}

        if len(blanket_basket)==0 or target_ not in blanket_basket:
            # print('len(blanket_basket)==0 or target not in blanket_basket')
            end = time.time()
            return [], {}, {}, end-begin

        
        cand_nodes = list(blanket_basket.keys())
        # print('cand_nodes are ')
        # print(cand_nodes)

        if g_orig.number_of_nodes()-len(cand_nodes)==0:
            g = g_orig
        else:
            g = g_orig.subgraph(cand_nodes)

        self.node_var = {node: self.vec2categ(self.sampled_data[node][:, blanket_basket[node]]) for node in blanket_basket} #after MLE sampling, only perturbed features are recorded in sampled data, so only features in basket are recorded.

        # print(self.node_var.keys())
        
        # print('Input g')
        # print(g.nodes)
        # print(' +++ cut nodes number: '+str(g_orig.number_of_nodes()-g.number_of_nodes()))

        print(f"start (target_) is: {target_}")
        S = [target_]
        raw_feature_exp = {}
        feats_to_pick, raw_feature_exp[target_] = self.explainFeature(target_, target_, blanket_basket[target_], S=[], p_threshold=p_threshold) # feats_to_pick is the raw pos of picked feats relative to basket

        if raw_feature_exp[target_] is None: # synthetic node overestimates feature importance
            print('target feature no exp')
            raw_feature_exp.pop(target_)
            # print('Feature exp invalid for node '+str(u))
            # break
            S.remove(target_)
            end = time.time()

            return list(S), {}, {}, end-begin

        else:
            # print(f'Fxplanation on {u}: ')
            # print(raw_feature_exp[u])
            self.sampled_data[target_] = self.sampled_data[target_][:, blanket_basket[target_][feats_to_pick]]

        nodes = [n for n in self.node_var if n!=target_]
        print('nodes')
        print(nodes)
        for u in nodes:

            pdData = self.cat_y_cap.astype(np.int8)

            for node in S: # conditioned on the current E
                cat = self.vec2categ(self.sampled_data[node]) # trick to reduce column number
                pdData = np.concatenate((pdData, cat), axis=1)


            pdData = np.concatenate((pdData, self.node_var[u]), axis=1) # compute dependency of synthetic node variable
                # print(pdData.shape)

            pdData = pd.DataFrame(pdData)

            ind_ori_to_sub = dict(zip(['target'] + S + [u], list(pdData.columns)))
            # print(ind_ori_to_sub.keys())

            p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[node_] for node_ in S], data=pdData, boolean=False)[1] for node in [u]]  
            
            if p_values[0]<p_threshold:
                feats_to_pick, raw_feature_exp[u] = self.explainFeature(target_, u, blanket_basket[u], S=S, p_threshold=p_threshold) # feats_to_pick is the raw pos of picked feats relative to basket

                if raw_feature_exp[u] is None: # synthetic node overestimates feature importance
                    raw_feature_exp.pop(u)
                    # I.add(u)
                    # print('Feature exp invalid for node '+str(u))
                    # break

                else:
                    # print(f'Fxplanation on {u}: ')
                    # print(raw_feature_exp[u])
                    self.sampled_data[u] = self.sampled_data[u][:, blanket_basket[u][feats_to_pick]]
                    S.append(u)

        if len(S)==0:
            # print('len(S)==0')
            end = time.time()

            return list(S), {}, {}, end-begin

        if len(S)>1:
            S = self.blanketShrinking(target_, g_orig, list(S), p_threshold)
            raw_feature_exp = {k: v for k, v in raw_feature_exp.items() if k in S}

        end = time.time()

        if self.dataset=='MUTAG':
            feature_exp = self.featureExpRaw2out(raw_feature_exp, target=target)
        else:
            feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping=mapping)

        # sparsity = {}
        # feature_exp = {}
        # if self.dataset=='DBLP':
        #     for node in raw_feature_exp:
        #         node_type = self.DBLP_idx2node_type[node%10]
        #         feature_exp[node] = self.computationalG_x_dict[node_type][
        #             mapping[node_type][node//10]].nonzero().cpu().numpy().T[0][raw_feature_exp[node]]

        # elif self.dataset=='IMDB':
        #     node_type = 'movie'
        #     for node in raw_feature_exp:
        #         feature_exp[node] = self.computationalG_x_dict[node_type][
        #             mapping[node_type][node]].nonzero().cpu().numpy().T[0][raw_feature_exp[node]]
        #         # sparsity[node] = raw_feature_exp[node].size/self.computationalG_x_dict[node_type][mapping[node]].count_nonzero().item()

        # print(' *** final output')
        # print(S)
        # print(raw_feature_exp)
        # print('time used: {:.2f}'.format(end-begin))

        return list(S), raw_feature_exp, feature_exp, end-begin

    # def blanketShrinking(self, target, g, S, p_threshold):
        
    #     print('Banket shrinking in topology space...')

    #     if self.dataset=='DBLP':
    #         target_ = target*10
    #     elif self.dataset=='IMDB':
    #         target_ = target

    #     pdData = self.cat_y_cap.astype(np.int8)

    #     can_remove = []
    #     for node in [n for n in S if n!=target_]:        
    #         subgraph = g.subgraph([n for n in S if n!=node])
    #         if nx.is_connected(subgraph):
    #             can_remove.append(node)
    #             # print(f"can remove {node}")

    #     for node in S:
    #         pdData = np.concatenate((pdData, self.node_var[node]), axis=1)
    #     # print(pdData.shape)
    #     print('can_remove')
    #     print(can_remove)

    #     pdData = pd.DataFrame(pdData)
    #     ind_ori_to_sub = dict(zip(['target'] + S, list(pdData.columns)))

    #     # p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[], data=pdData, boolean=False)[1] for node in S]
    #     p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[], data=pdData, boolean=False)[1] for node in can_remove]
    #     print(p_values)

    #     selected_nodes = set(S)
    #     for idx in np.argsort(-1*np.asarray(p_values)): # trick to reduce conditional independency test, start from most independent variable (largest p value)
    #         # print(idx)
    #         node = can_remove[idx]
    #         # if chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[x] for x in selected_nodes if x!=node], data=pdData, boolean=True, significance_level=p_threshold):
    #         if g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[x] for x in selected_nodes if x!=node], data=pdData, boolean=True, significance_level=p_threshold):

    #             # cand_nodes = selected_nodes.copy()
    #             # cand_nodes.remove(node)
    #             # cands_subgraph = g.subgraph(cand_nodes)
    #             # if not nx.is_connected(cands_subgraph):
    #             #     # print(f"Should not remove {node}")
    #             #     continue
    #             # else:
    #             #     print(str(node)+' should be removed.')
    #             print(str(node)+' should be removed.')
    #             selected_nodes.remove(node)

    #     if len(selected_nodes)==len(S):
    #         print('No variable is removed')

    #     return list(selected_nodes)

    def blanketShrinking(self, target_, g, S, p_threshold):
        
        # print('Banket shrinking in topology space...')

        # if self.dataset=='DBLP':
        #     target_ = target*10
        # elif self.dataset=='IMDB':
        #     target_ = target

        pdData = self.cat_y_cap.astype(np.int8)

        can_remove = []
        for node in [n for n in S if n!=target_]:        
            subgraph = g.subgraph([n for n in S if n!=node])
            if nx.is_connected(subgraph):
                can_remove.append(node)
                # print(f"can remove {node}")

        for node in S:
            pdData = np.concatenate((pdData, self.vec2categ(self.sampled_data[node])), axis=1)

        for node in S:
            pdData = np.concatenate((pdData, self.node_var[node]), axis=1)

        # print(pdData.shape)
        # print('can_remove')
        # print(can_remove)

        pdData = pd.DataFrame(pdData)
        ind_ori_to_sub = dict(zip(['target'] + S + ['syn'+str(n) for n in S], list(pdData.columns)))

        # p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[], data=pdData, boolean=False)[1] for node in S]
        p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[], data=pdData, boolean=False)[1] for node in can_remove]
        # print(p_values)

        selected_nodes = set(S)
        for idx in np.argsort(-1*np.asarray(p_values)): # trick to reduce conditional independency test, start from most independent variable (largest p value)
            # print(idx)
            node = can_remove[idx]
            # if chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[x] for x in selected_nodes if x!=node], data=pdData, boolean=True, significance_level=p_threshold):
            if g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub['syn'+str(node)], Z=[ind_ori_to_sub[n] for n in selected_nodes if n!=node], data=pdData, boolean=True, significance_level=p_threshold):

                # cand_nodes = selected_nodes.copy()
                # cand_nodes.remove(node)
                # cands_subgraph = g.subgraph(cand_nodes)
                # if not nx.is_connected(cands_subgraph):
                #     # print(f"Should not remove {node}")
                #     continue
                # else:
                #     print(str(node)+' should be removed.')
                # print(str(node)+' should be removed.')
                selected_nodes.remove(node)

        # if len(selected_nodes)==len(S):
        #     print('No variable is removed')
        return list(selected_nodes)

    # def blanketShrinkingVec2Cat(self, g, S, p_threshold):
        
    #     print('Banket shrinking in topology space...')

    #     pdData = self.cat_y_cap.astype(np.int8)

    #     for node in S:
    #         cat = self.vec2categ(self.sampled_data[node])
    #         pdData = np.concatenate((pdData, cat), axis=1)
    #     # print(pdData.shape)

    #     pdData = pd.DataFrame(pdData)
    #     ind_ori_to_sub = dict(zip(['target'] + S, list(pdData.columns)))

    #     # p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[], data=pdData, boolean=False)[1] for node in S]
    #     p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[], data=pdData, boolean=False)[1] for node in S]

    #     selected_nodes = set(S)
    #     for idx in np.argsort(-1*np.asarray(p_values)): # trick to reduce conditional independency test, start from most independent variable (largest p value)
    #         # print(idx)
    #         node = S[idx]
    #         # if chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[x] for x in selected_nodes if x!=node], data=pdData, boolean=True, significance_level=p_threshold):
    #         if g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node], Z=[ind_ori_to_sub[x] for x in selected_nodes if x!=node], data=pdData, boolean=True, significance_level=p_threshold):

    #             cand_nodes = selected_nodes.copy()
    #             cand_nodes.remove(node)
    #             cands_subgraph = g.subgraph(cand_nodes)
    #             if not nx.is_connected(cands_subgraph):
    #                 # print(f"Should not remove {node}")
    #                 continue
    #             else:
    #                 print(str(node)+' should be removed.')
    #                 selected_nodes.remove(node)

    #     if len(selected_nodes)==len(S):
    #         print('No variable is removed')

    #     return list(selected_nodes)
        

    def basketSearching(self, target,  n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, p_threshold=0.05, pred_threshold=0.01):

        # begin = time.time()

        # self.computationalG_x_dict = None
        # self.computationalG_edge_index_dict = None

        # mapping, _, g = self.__subgraph__(target) 

        # # print(self.computationalG_x_dict['author'].device)

        # self.computationalG2device()
        # # print(self.computationalG_x_dict['author'].device)

        # self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

        basket = {}

        if self.dataset=='DBLP' or self.dataset=='IMDB':
            if self.dataset=='DBLP':
                _, _, g = self.__subgraph__(target) 
                target_ = target*10
                # _, _ = self.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
            elif self.dataset=='IMDB':
                _, _, g = self.__subgraph__(target) 
                target_ = target
                # _, _ = self.uniformPerturbIMDB(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)


            U = set([target_])
            processed = set()
            p_values = [-1]
            c = 0

            while len(U)>0:

                c+=1

                # print(' --- Round '+str(c))
                # print(U)
                current_U = U.copy()
                for u in current_U:

                    S_data = np.concatenate((self.cat_y_cap.astype(np.int8), self.sampled_data[u]), axis=1)

                    pdData = pd.DataFrame(S_data)
                    ind_ori_to_sub = dict(zip(['target'] + ['f'+str(i) for i in range(self.sampled_data[u].shape[1])], list(pdData.columns)))

                    # feat_p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in range(self.sampled_data[u].shape[1])]]
                    feat_p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in range(self.sampled_data[u].shape[1])]]

                    feats_to_pick = [i for i, x in enumerate(feat_p_values) if x < p_threshold]

                    if len(feats_to_pick)>0:
                        # print(f' Basket searching: {u} is processed.')
                        basket[u] = np.array(feats_to_pick)
                        # print(feat_p_values)
                        # print(feats_to_pick)
                        U = U.union(set(g[u]))

                    elif len(feats_to_pick)==0 and u==target_:
                        return {}
                        # raise ValueError("no feature dependent on target")

                processed = processed.union(current_U)
                U = U.difference(processed)
            return basket

        elif self.dataset=='MUTAG':
            
            lowest_p = 1
            start = 0

            for u in range(self.MUTAG_dataset[target].x.shape[0]):

                S_data = np.concatenate((self.cat_y_cap.astype(np.int8), self.sampled_data[u]), axis=1)

                pdData = pd.DataFrame(S_data)
                ind_ori_to_sub = dict(zip(['target'] + ['f'+str(i) for i in range(self.sampled_data[u].shape[1])], list(pdData.columns)))

                feat_p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in range(self.sampled_data[u].shape[1])]]

                feats_to_pick = [i for i, x in enumerate(feat_p_values) if x < p_threshold]

                if len(feats_to_pick)>0:
                    # print(f' Basket searching: {u} is processed.')
                    basket[u] = np.array(feats_to_pick)
                    # print(feat_p_values)
                    # print(feats_to_pick)
                    if min(feat_p_values)<lowest_p:
                        lowest_p = min(feat_p_values)
                        start = u

            g = to_networkx(self.MUTAG_dataset[target], to_undirected=True)
            basket_subgraph = g.subgraph(list(basket.keys()))
            for c in nx.connected_components(basket_subgraph):
                if start in c:
                    basket = {k:v for k, v in basket.items() if k in c}
                    break
        
        # end = time.time()

            return basket, start

    # def exhaustedBasketSearching(self, target,  n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, p_threshold=0.05, pred_threshold=0.01):

    #     # begin = time.time()

    #     self.computationalG_x_dict = None
    #     self.computationalG_edge_index_dict = None

    #     mapping, _, g = self.__subgraph__(target) 

    #     # print(self.computationalG_x_dict['author'].device)

    #     self.computationalG2device()
    #     # print(self.computationalG_x_dict['author'].device)

    #     self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

    #     if self.dataset=='DBLP':
    #         _, _ = self.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
    #     elif self.dataset=='IMDB':
    #         _, _ = self.uniformPerturbIMDB(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

    #     if self.dataset=='DBLP':
    #         target_ = target*10
    #     elif self.dataset=='IMDB':
    #         target_ = target

    #     raw_feature_exp = {}
    #     for u in g.nodes:
    #         S_data = np.concatenate((self.cat_y_cap.astype(np.int8), self.sampled_data[u]), axis=1)

    #         pdData = pd.DataFrame(S_data)
    #         ind_ori_to_sub = dict(zip(['target'] + ['f'+str(i) for i in range(self.sampled_data[u].shape[1])], list(pdData.columns)))

    #         # feat_p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in range(self.sampled_data[u].shape[1])]]
    #         feat_p_values = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[feat_], Z=[], data=pdData, boolean=False)[1] for feat_ in ['f'+str(i) for i in range(self.sampled_data[u].shape[1])]]

    #         feats_to_pick = [i for i, x in enumerate(feat_p_values) if x < p_threshold]

    #         raw_feature_exp[u] = feats_to_pick

    #         print(feat_p_values)
    #         print(feats_to_pick)

    #         if len(feats_to_pick)==0 and u==target_:
    #             raise ValueError("no feature dependent on target")

    #     # end = time.time()

    #     return raw_feature_exp

    # def factualMLE(self, target, S, raw_feature_exp, n_cat_value=3, num_samples=800, k=20, p_perturb=.5, pred_threshold=.01):

    #     # if len(raw_feature_exp)==0:
    #     #     return S, raw_feature_exp, S, raw_feature_exp

    #     self.computationalG_x_dict = None
    #     self.computationalG_edge_index_dict = None

    #     # if self.dataset=='DBLP':
    #     #     _, mapping = self.MLEsamplingDBLP(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=True)

    #     # elif self.dataset=='IMDB':
    #     #     _, mapping = self.MLEsamplingIMDB(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=True)

    #     mapping, _, _ = self.__subgraph__(target)

    #     feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping)

    #     val = 2
    #     factual_feat_exp = {}
    #     factual_S = []

    #     for node in S:
    #         sampled_data = self.sampled_data[node][np.any(1-self.sampled_data[node],axis=1), :] #at least one feature is NOT perturbed
    #         cat_y_cap = self.cat_y_cap[np.any(1-self.sampled_data[node],axis=1), :] #at least one feature is NOT perturbed

    #         cat = self.vec2categ(sampled_data)
    #         cat_dict = dict(zip(np.unique(cat, return_counts=True)[0], np.unique(cat, return_counts=True)[1]))

    #         occ = cat[np.nonzero(cat_y_cap)[0]]
    #         occ_dict = dict(zip(np.unique(occ, return_counts=True)[0], np.unique(occ, return_counts=True)[1]))

    #         par = {cat_value:occ_dict[cat_value]/cat_dict[cat_value] for cat_value in occ_dict}
    #         # print(par)
    #         # highest_par = {x:np.count_nonzero(sampled_data[(cat==x).nonzero()[0]][0]) for x in par if par[x]==max(list(par.values()))}
    #         if node in feature_exp:
    #             highest_par = {x:feature_exp[node][(1-sampled_data[(cat==x).nonzero()[0]][0]).nonzero()[0]] for x in par if par[x]==max(list(par.values())) and np.count_nonzero(1-sampled_data[(cat==x).nonzero()[0]][0])>0}
    #             if len(highest_par)==0:
    #                 # print('No value found!')
    #                 pass

    #             elif len(highest_par)==1:
    #                 pick = list(highest_par.keys())[0]
    #                 factual_feat_exp[node] = feature_exp[node][(1-sampled_data[(cat==pick).nonzero()[0]][0]).nonzero()[0]]
    #                 factual_S.append(node)

    #             elif len(highest_par)>1:
    #                 feats = []
    #                 for cat_val in highest_par:
    #                     feats+=highest_par[cat_val].tolist()
    #                 feat_weight = {x:feats.count(x) for x in feats}
    #                 weight = {cat_val:sum([feat_weight[x] for x in highest_par[cat_val]]) for cat_val in highest_par}
    #                 pick = max(weight, key=weight.get)
    #                 # print(pick)
    #                 factual_S.append(node)
    #                 factual_feat_exp[node] = feature_exp[node][(1-sampled_data[(cat==pick).nonzero()[0]][0]).nonzero()[0]]
    #                 # print(counterfactual_feat_exp[node])
    #             # print(self.calFidelity(target, S, counterfactual_feat_exp)[:4])
    #             # print()
    #         else:
    #             if max(par, key=par.get)==0:
    #                 factual_S.append(node)

    #     # if len(factual_S)==0:
    #     #     factual_S = S
    #     #     factual_feat_exp = feature_exp
    #     # print(self.calFidelity(target, factual_S, factual_feat_exp)[:4])

    #     return factual_S, factual_feat_exp

    # def counterfactualMLE(self, target, S, raw_feature_exp, n_cat_value=3, num_samples=1000, k=20, p_perturb=.5, pred_threshold=.01):

    #     # if len(raw_feature_exp)==0:
    #     #     return S, raw_feature_exp, S, raw_feature_exp

    #     self.computationalG_x_dict = None
    #     self.computationalG_edge_index_dict = None

    #     # if self.dataset=='DBLP':
    #     #     _, mapping = self.MLEsamplingDBLP(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

    #     # elif self.dataset=='IMDB':
    #     #     _, mapping = self.MLEsamplingIMDB(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

    #     mapping, _, _ = self.__subgraph__(target)

    #     feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping)

    #     val = 0 # counterfacture
    #     counterfactual_feat_exp = {}
    #     counterfactual_S = []


    #     for node in S:
    #         sampled_data = self.sampled_data[node][np.any(self.sampled_data[node],axis=1), :] #at least one feature is perturbed
    #         cat_y_cap = self.cat_y_cap[np.any(self.sampled_data[node],axis=1), :]
    #         cat = self.vec2categ(sampled_data)
    #         cat_dict = dict(zip(np.unique(cat, return_counts=True)[0], np.unique(cat, return_counts=True)[1]))

    #         occ = cat[np.nonzero(cat_y_cap==val)[0]]
    #         occ_dict = dict(zip(np.unique(occ, return_counts=True)[0], np.unique(occ, return_counts=True)[1]))

    #         # par = {cat_value:occ_dict[cat_value]/cat_dict[cat_value] for cat_value in occ_dict if cat_value!=0}
    #         par = {cat_value:occ_dict[cat_value]/cat_dict[cat_value] for cat_value in occ_dict}
    #         # print(par)
    #         # highest_par = {x:np.count_nonzero(sampled_data[(cat==x).nonzero()[0]][0]) for x in par if par[x]==max(list(par.values()))}
    #         if node in feature_exp:
    #             highest_par = {x:feature_exp[node][sampled_data[(cat==x).nonzero()[0]][0].nonzero()[0]] for x in par if par[x]==max(list(par.values())) and np.count_nonzero(sampled_data[(cat==x).nonzero()[0]][0])>0}
    #             if len(highest_par)==0:
    #                 pass
    #                 # print('No value found!')

    #             elif len(highest_par)==1:
    #                 pick = list(highest_par.keys())[0]
    #                 counterfactual_feat_exp[node] = feature_exp[node][sampled_data[(cat==pick).nonzero()[0]][0].nonzero()[0]]
    #                 counterfactual_S.append(node)

    #             elif len(highest_par)>1:
    #                 feats = []
    #                 for cat_val in highest_par:
    #                     feats+=highest_par[cat_val].tolist()
    #                 feat_weight = {x:feats.count(x) for x in feats}
    #                 weight = {cat_val:sum([feat_weight[x] for x in highest_par[cat_val]]) for cat_val in highest_par}
    #                 pick = max(weight, key=weight.get)
    #                 # print(pick)
    #                 counterfactual_S.append(node)
    #                 counterfactual_feat_exp[node] = feature_exp[node][sampled_data[(cat==pick).nonzero()[0]][0].nonzero()[0]]
    #                 # print(counterfactual_feat_exp[node])
    #             # print(self.calFidelity(target, S, counterfactual_feat_exp)[:4])
    #             # print()
    #         else: # node type is conference, no feature exp
    #             if max(par, key=par.get)==1:
    #                 counterfactual_S.append(node)

    #     # if len(counterfactual_S)==0:
    #     #     counterfactual_S = S
    #     #     counterfactual_feat_exp = feature_exp
    #     # print(self.calFidelity(target, counterfactual_S, counterfactual_feat_exp)[:4])

    #     return counterfactual_S, counterfactual_feat_exp

    # def MLE(self, target, S, raw_feature_exp, n_cat_value=3, num_samples=1000, k=20, p_perturb=.5, pred_threshold=.01):

    #     # if len(raw_feature_exp)==0:
    #     #     return S, raw_feature_exp, S, raw_feature_exp

    #     self.computationalG_x_dict = None
    #     self.computationalG_edge_index_dict = None

    #     # if self.dataset=='DBLP':
    #     #     _, mapping = self.MLEsamplingDBLP(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

    #     # elif self.dataset=='IMDB':
    #     #     _, mapping = self.MLEsamplingIMDB(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

    #     mapping, _, _ = self.__subgraph__(target)
    #     feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping)

    #     counterfactual_feat_exp = {}
    #     factual_feat_exp = {}
    #     for node in feature_exp:
    #         counterF_par = np.sum(self.sampled_data[node][np.nonzero(self.cat_y_cap==0)[0],:], axis=0)/np.count_nonzero(self.cat_y_cap==0)
    #         # counterF_par = np.sum(self.sampled_data[node][np.nonzero(self.cat_y_cap==0)[0],:], axis=0)/np.sum(self.sampled_data[node], axis=0)
    #         counterfactual_feat_exp[node] = feature_exp[node][np.nonzero(counterF_par>0.5)[0]]

    #         # F_par = np.sum(self.sampled_data[node][np.nonzero(self.cat_y_cap==2)[0],:], axis=0)/np.sum(self.sampled_data[node], axis=0)
    #         F_par = np.sum(1-self.sampled_data[node][np.nonzero(self.cat_y_cap==2)[0],:], axis=0)/np.count_nonzero(self.cat_y_cap==2)
    #         factual_feat_exp[node] = feature_exp[node][np.nonzero(F_par>0.5)[0]]

    #     return counterfactual_S, counterfactual_feat_exp

    def factual_synMLE(self, target, S, raw_feature_exp, n_cat_value=3,amplifier=5, num_samples=1000, k=20, p_perturb=.5, pred_threshold=.0000001):

        # if len(raw_feature_exp)==0:
        #     return S, raw_feature_exp, S, raw_feature_exp

        self.computationalG_x_dict = None
        self.computationalG_edge_index_dict = None

        # k *= amplifier
        # num_samples *= amplifier
        # print(f"amplified k: {k}")

        # factual
        if self.dataset=='DBLP':
            mapping, _, _ = self.__subgraph__(target)
            feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping=mapping)

            _, _ = self.MLEsamplingDBLP(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=True)

        elif self.dataset=='IMDB':
            mapping, _, _ = self.__subgraph__(target)
            feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping=mapping)
            _, _ = self.MLEsamplingIMDB(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=True)
            # _, _ = self.MLEsamplingIMDB(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

        elif self.dataset=='MUTAG':
            feature_exp = self.featureExpRaw2out(raw_feature_exp, target=target)
            self.MLEsamplingMUTAG(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=True)
            # self.MLEsamplingMUTAG(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

            # self.uniformPerturbMUTAG(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            # self.MLE_sampling_fail = False
            # self.MLE_sampled_data = self.sampled_data
            # self.MLE_cat_y_cap = self.cat_y_cap
            # self.MLE_ori_sampled_y_cap = self.ori_sampled_y_cap


        if self.MLE_sampling_fail:
            print('MLE sampling failed!')
            if self.dataset=='DBLP':
                _, _ = self.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
            elif self.dataset=='IMDB':
                _, _ = self.uniformPerturbIMDB(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
            elif self.dataset=='MUTAG':
                self.uniformPerturbMUTAG(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            self.MLE_sampled_data = self.sampled_data
            self.MLE_cat_y_cap = self.cat_y_cap
            self.MLE_ori_sampled_y_cap = self.ori_sampled_y_cap

        invalid_samples = set(range(self.MLE_sampled_data[S[0]].shape[0]))
        for node in S:
            invalid_samples = invalid_samples.intersection(set(np.nonzero(self.MLE_sampled_data[node].sum(axis=-1)==self.MLE_sampled_data[node].shape[-1])[0].tolist()
        ))

        if len(invalid_samples)!=0:
            self.MLE_cat_y_cap[list(invalid_samples)]=0

        factual_feat_exp = {}

        if len(S)==1:
            values, counts = np.unique(self.MLE_sampled_data[S[0]][(self.MLE_cat_y_cap == 2).nonzero()[0],:], axis=0, return_counts=True)
            if np.count_nonzero(counts == counts.max()) > 1:
                opt_score = 0
                cand_values = values[counts == counts.max()]
                for i in range(cand_values.shape[0]):
                    score = self.MLE_ori_sampled_y_cap[(self.MLE_sampled_data[S[0]]==cand_values[i, :]).all(axis=1)].max()
                    if score > opt_score:
                        opt_score = score
                # for node in S:
                #     factual_feat_exp[node] = feature_exp[node][self.sampled_data[node][(self.ori_sampled_y_cap==opt_score).nonzero()[0],:][0]]
                # factual_S = list(factual_feat_exp.keys())
            else:
                cand_value = values[counts == counts.max()]
                opt_score = self.MLE_ori_sampled_y_cap[(self.MLE_sampled_data[S[0]]==cand_value).all(axis=1)].max()
        else:
            par_table = self.MLE_cat_y_cap.astype(np.int8)
            for node in S:
                # pdData = np.concatenate((pdData, bool_samples[node]), axis=1)
                par_table = np.concatenate((par_table, self.vec2categ(self.MLE_sampled_data[node])), axis=1)

            values, counts = np.unique(par_table[(self.MLE_cat_y_cap==2).nonzero()[0],1:], axis=0, return_counts=True)
            if np.count_nonzero(counts==counts.max())>1:
                opt_score = 0
                cand_values = values[counts==counts.max()]
                for i in range(cand_values.shape[0]):
                    score = self.MLE_ori_sampled_y_cap[(par_table[:, 1:]==cand_values[i, :]).all(axis=1)].max()
                    if score>opt_score:
                        opt_score=score
                # for node in S:
                #     factual_feat_exp[node] = feature_exp[node][self.sampled_data[node][(self.ori_sampled_y_cap==opt_score).nonzero()[0],:][0]]
                # factual_S = list(factual_feat_exp.keys())
            else:
                cand_value = values[counts==counts.max()]
                opt_score = self.MLE_ori_sampled_y_cap[(par_table[:, 1:]==cand_value).all(axis=1)].max()

        # if np.count_nonzero(self.ori_sampled_y_cap==opt_score)>0:
        pos = (self.MLE_ori_sampled_y_cap==opt_score).nonzero()[0][0]

        for node in S:
            # factual_feat_exp[node] = feature_exp[node][self.sampled_data[node][(self.ori_sampled_y_cap==opt_score).nonzero()[0],:][0]]
            factual_feat_exp[node] = feature_exp[node][self.MLE_sampled_data[node][pos,:]==0]
        factual_S = list(factual_feat_exp.keys())

        # if self.dataset=='MUTAG':
        # print('cut empty nodes in exp for mutag dataset')
        factual_S = [k for k in factual_feat_exp if factual_feat_exp[k].size>0]
        factual_feat_exp = {k:v for k,v in factual_feat_exp.items() if v.size>0}

        return factual_S, factual_feat_exp

    def counterfactual_synMLE(self, target, S, raw_feature_exp, amplifier=5,n_cat_value=3, num_samples=1000, k=20, p_perturb=.5, pred_threshold=.0000001,):

        # if len(raw_feature_exp)==0:
        #     return S, raw_feature_exp, S, raw_feature_exp

        self.computationalG_x_dict = None
        self.computationalG_edge_index_dict = None

        # k *= amplifier
        # num_samples *= amplifier

        # print(f"amplified k: {k}")

        # counterfactual
        if self.dataset=='DBLP':
            mapping, _, _ = self.__subgraph__(target)
            feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping=mapping)

            _, _ = self.MLEsamplingDBLP(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

        elif self.dataset=='IMDB':
            mapping, _, _ = self.__subgraph__(target)
            feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping=mapping)
            _, _ = self.MLEsamplingIMDB(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

        elif self.dataset=='MUTAG':
            feature_exp = self.featureExpRaw2out(raw_feature_exp, target=target)
            self.MLEsamplingMUTAG(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

        if self.MLE_sampling_fail:
            print('MLE sampling failed!')
            if self.dataset=='DBLP':
                _, _ = self.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
            elif self.dataset=='IMDB':
                _, _ = self.uniformPerturbIMDB(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)
            elif self.dataset=='MUTAG':
                self.uniformPerturbMUTAG(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

            self.MLE_sampled_data = self.sampled_data
            self.MLE_cat_y_cap = self.cat_y_cap
            self.MLE_ori_sampled_y_cap = self.ori_sampled_y_cap

        invalid_samples = set(range(self.MLE_sampled_data[S[0]].shape[0]))
        for node in S:
            invalid_samples = invalid_samples.intersection(set(np.nonzero(self.MLE_sampled_data[node].sum(axis=-1)==0)[0].tolist()
        ))
        if len(invalid_samples)!=0:
            self.MLE_cat_y_cap[list(invalid_samples)] = 1

        # mapping, _, _ = self.__subgraph__(target)
        #
        # feature_exp = self.featureExpRaw2out(raw_feature_exp, mapping)
        # # counterfactual
        # if self.dataset=='DBLP':
        #     _, mapping = self.MLEsamplingDBLP(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

        # elif self.dataset=='IMDB':
        #     _, mapping = self.MLEsamplingIMDB(target, S=S, raw_feature_exp=raw_feature_exp, num_samples=num_samples, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold, factual=False)

        # if self.MLE_sampling_fail and self.dataset=='DBLP':
        #     print('MLE sampling failed!')
        #     _, _ = self.uniformPerturbDBLP(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

        #     self.MLE_sampled_data = self.sampled_data
        #     self.MLE_cat_y_cap = self.cat_y_cap
        #     self.MLE_ori_sampled_y_cap = self.ori_sampled_y_cap

        # elif self.MLE_sampling_fail and self.dataset=='IMDB':
        #     print('MLE sampling failed!')
        #     _, _ = self.uniformPerturbIMDB(target, k=k, p_perturb=p_perturb, pred_threshold=pred_threshold)

        #     self.MLE_sampled_data = self.sampled_data
        #     self.MLE_cat_y_cap = self.cat_y_cap
        #     self.MLE_ori_sampled_y_cap = self.ori_sampled_y_cap


        counterfactual_feat_exp = {}

        if len(S)==1:
            values, counts = np.unique(self.MLE_sampled_data[S[0]][(self.MLE_cat_y_cap==0).nonzero()[0],:], axis=0, return_counts=True)
            if np.count_nonzero(counts == counts.max()) > 1:
                opt_score = 1
                cand_values = values[counts == counts.max()]
                for i in range(cand_values.shape[0]):
                    score = self.MLE_ori_sampled_y_cap[(self.MLE_sampled_data[S[0]]==cand_values[i, :]).all(axis=1)].max()
                    if score < opt_score:
                        opt_score = score
                # for node in S:
                #     factual_feat_exp[node] = feature_exp[node][self.sampled_data[node][(self.ori_sampled_y_cap==opt_score).nonzero()[0],:][0]]
                # factual_S = list(factual_feat_exp.keys())
            else:
                cand_value = values[counts == counts.max()]
                opt_score = self.MLE_ori_sampled_y_cap[(self.MLE_sampled_data[S[0]]==cand_value).all(axis=1)].max()

        else:
            par_table = self.MLE_cat_y_cap.astype(np.int8)
            for node in S:
                # pdData = np.concatenate((pdData, bool_samples[node]), axis=1)
                par_table = np.concatenate((par_table, self.vec2categ(self.MLE_sampled_data[node])), axis=1)

            values, counts = np.unique(par_table[(self.MLE_cat_y_cap==0).nonzero()[0],1:], axis=0, return_counts=True)
            if np.count_nonzero(counts==counts.max())>1:
                opt_score = 1
                cand_values = values[counts==counts.max()]
                for i in range(cand_values.shape[0]):
                    score = self.MLE_ori_sampled_y_cap[(par_table[:, 1:]==cand_values[i, :]).all(axis=1)].min()
                    if score<opt_score:
                        opt_score=score
                # for node in S:
                #     factual_feat_exp[node] = feature_exp[node][self.sampled_data[node][(self.ori_sampled_y_cap==opt_score).nonzero()[0],:][0]]
                # factual_S = list(factual_feat_exp.keys())
            else:
                cand_value = values[counts==counts.max()]
                opt_score = self.MLE_ori_sampled_y_cap[(par_table[:, 1:]==cand_value).all(axis=1)].min()

        pos = (self.MLE_ori_sampled_y_cap==opt_score).nonzero()[0][0]

        for node in S:
            counterfactual_feat_exp[node] = feature_exp[node][self.MLE_sampled_data[node][pos,:]==1]
        # counterfactual_S = list(counterfactual_feat_exp.keys())

        # if self.dataset=='MUTAG':
        # print('cut empty nodes in exp for mutag dataset')
        counterfactual_S = [k for k in counterfactual_feat_exp if counterfactual_feat_exp[k].size>0]
        counterfactual_feat_exp = {k:v for k,v in counterfactual_feat_exp.items() if v.size>0}

        return counterfactual_S, counterfactual_feat_exp


    def explainDBLP_pale(self, target, num_samples=1000, n_cat_value=10, p_threshold = 0.05, p_perturb=0.5, re_sample = True):
    
        print('Number of samples required is ' +str(num_samples))
        if re_sample:
            g, mapping = self.dataGenDBLP(target, num_samples = num_samples, top_node = None, p_perturb = p_perturb, explain_pale=True)
        else:
            mapping, _, g = self.__subgraph__(target)

        ### turn the y cap into catigorical variable with 5 different values
        # max_drop = self.orig_pred - self.sampled_y_cap.min()
        perturb_range = self.sampled_y_cap.max() - self.sampled_y_cap.min()
        # if max_drop==0:
        if perturb_range==0:
            print('GNN prediction never change! \n')
            return [target*10]
            
        # delta = max_drop/5
        delta = perturb_range/n_cat_value
        tmp = (self.orig_pred - self.sampled_y_cap)/delta
        if tmp.min()<0:
            cat_y_cap = np.ceil(tmp - tmp.min())
        else:
            cat_y_cap = np.ceil(tmp)
        ###

        S = set()
        U = [target*10]
        p_values=[-1]

        c=0

        while min(p_values) < p_threshold:

            nodes_to_pick = [U[i] for i, x in enumerate(p_values) if x == min(p_values)]
            # u = U[p_values.index(min(p_values))]
            # print(' --- Round '+str(c) + ', picked '+str(u))
            print(' --- Round '+str(c) + ', picked ')
            print(nodes_to_pick)

            U = set(U)
            S = set(S)
            for u in nodes_to_pick:
                U.remove(u)
                S.add(u)
                U = U.union(set(g[u]).difference(S))

            if len(U)==0:
                break

            U = list(U) # use list to fix order
            S = list(S)
            print(S)
            print(U)

            pdData = cat_y_cap.astype(np.int8)

            for node in S+U:
                # pdData = np.concatenate((pdData, bool_samples[node]), axis=1)
                pdData = np.concatenate((pdData, self.sampled_data[node]), axis=1)
            # print(pdData.shape)

            pdData = pd.DataFrame(pdData)

            # ind_sub_to_ori = dict(zip(list(pdData.columns), ['target']+S+U))
            ind_ori_to_sub = dict(zip(['target']+S+U, list(pdData.columns)))

            # if c==0 and (cat_y_cap==self.sampled_data[target*10]).all():
            #     print('DoF is 0')
            #     p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node_], Z=[], data=pdData, boolean=False)[1] for node_ in U]
            # else:
            #     p_values = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node_], Z=[ind_ori_to_sub[n] for n in S], data=pdData, boolean=False)[1] for node_ in U]

            # chi_test = [chi_square(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node_], Z=[ind_ori_to_sub[n] for n in S], data=pdData, boolean=False) for node_ in U]
            chi_test = [g_sq(X=ind_ori_to_sub['target'], Y=ind_ori_to_sub[node_], Z=[ind_ori_to_sub[n] for n in S], data=pdData, boolean=False) for node_ in U]
            p_values = [x[1] for x in chi_test]

            if min([x[-1] for x in chi_test])==0:
                print('DoF is 0')
                print(p_values)
                break

            # p_values = [chi_square(Y=ind_ori_to_sub['target'], X=ind_ori_to_sub[node_], Z=[ind_ori_to_sub[n] for n in S], data=pdData, boolean=False)[1] for node_ in U]
            print(p_values)

            c+=1
        print(' *** final output')
        print(S)
        print()

        return S

    def computationalG2device(self):
        # print(self.device)
        # for key in self.data.edge_index_dict:
        #     self.data.edge_index_dict[key].to(self.device)
        #     print(self.data.edge_index_dict[key].device)
        if self.dataset=='DBLP':
            node_type = 'author'
            edge_type = ('author','to','paper')
        elif self.dataset=='IMDB':
            node_type = 'movie'
            edge_type =('movie', 'metapath_0', 'movie')

        if self.computationalG_x_dict[node_type].device != self.device:
            for nt in self.computationalG_x_dict:
                self.computationalG_x_dict[nt]=self.computationalG_x_dict[nt].to(self.device)

        if self.computationalG_edge_index_dict[edge_type].device!=self.device:
            for edge in self.computationalG_edge_index_dict:
                self.computationalG_edge_index_dict[edge] = self.computationalG_edge_index_dict[edge].to(self.device)

    def eval_exp_pale(self, target, explanation, g, mapping):

        self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

        fidelity_x_dict = copy.deepcopy(self.computationalG_x_dict)
        for node in explanation:
            fidelity_x_dict[self.DBLP_idx2node_type[node%10]][mapping[self.DBLP_idx2node_type[node%10]][node//10]]=0

        infidelity_x_dict = copy.deepcopy(self.computationalG_x_dict)
        for node in set(g.nodes).difference(explanation):
            infidelity_x_dict[self.DBLP_idx2node_type[node%10]][mapping[self.DBLP_idx2node_type[node%10]][node//10]]=0

        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            fidelity_out = nn.functional.softmax(self.model(fidelity_x_dict, self.computationalG_edge_index_dict), dim=-1)[mapping['author'][target]]
            infidelity_out = nn.functional.softmax(self.model(infidelity_x_dict, self.computationalG_edge_index_dict), dim=-1)[mapping['author'][target]]

        fidelity = self.orig_pred - fidelity_out[self.orig_pred_label].item()
        explanation_score = infidelity_out[self.orig_pred_label].item()
        infidelity = self.orig_pred - explanation_score

        return fidelity, infidelity, explanation_score, len(explanation)/g.number_of_nodes()

    def calFidelity(self, target, S, feature_exp, evaluator=False):
        # S is the set of nodes in the explanation
        # feature_exp is a dict of feature explanation

        if len(S)==0:
            print('Empty explanation!')
            return

        if evaluator:
            self.computationalG_x_dict = None
            self.computationalG_edge_index_dict = None

            # for module in self.model.modules():
            #     if isinstance(module, MessagePassing):
            #         module.__explain_hgnn__ = False # for gnnexplainer
        

        mapping, reversed_mapping, g = self.__subgraph__(target)

        self.orig_pred, self.orig_pred_label = self.getOrigPred(target, mapping)

        fidelity_x_dict = copy.deepcopy(self.computationalG_x_dict)

        for node in S:
            if self.dataset=='DBLP':
                # if node%10!=3:
                if node%10!=3 and node not in feature_exp: # some node does not have nonzero features, then leave it
                    continue
                elif node in feature_exp:
                    fidelity_x_dict[self.DBLP_idx2node_type[node%10]][mapping[self.DBLP_idx2node_type[node%10]][node//10]][feature_exp[node]]=0
                else:
                    fidelity_x_dict[self.DBLP_idx2node_type[node%10]][mapping[self.DBLP_idx2node_type[node%10]][node//10]]=0
            elif self.dataset=='IMDB':
                if node in feature_exp:
                    fidelity_x_dict['movie'][mapping['movie'][node]][feature_exp[node]]=0
                else:
                    continue

        infidelity_x_dict = {k:torch.zeros_like(self.computationalG_x_dict[k]) for k in self.computationalG_x_dict}
        for node in S:
            if self.dataset=='DBLP':
                if node%10!=3 and node not in feature_exp: # some node does not have nonzero features, then leave it
                    continue
                # if node%10!=3:
                elif node in feature_exp:
                    infidelity_x_dict[self.DBLP_idx2node_type[node%10]][mapping[self.DBLP_idx2node_type[node%10]][node//10]][feature_exp[node]]=1
                else:
                    infidelity_x_dict[self.DBLP_idx2node_type[node%10]][mapping[self.DBLP_idx2node_type[node%10]][node//10]]=1
            elif self.dataset=='IMDB':
                if node in feature_exp:
                    infidelity_x_dict['movie'][mapping['movie'][node]][feature_exp[node]]=1
                else:
                    continue

        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            if self.dataset=='DBLP':
                fidelity_out = nn.functional.softmax(self.model(fidelity_x_dict, self.computationalG_edge_index_dict), dim=-1)[mapping['author'][target]]
                infidelity_out = nn.functional.softmax(self.model(infidelity_x_dict, self.computationalG_edge_index_dict), dim=-1)[mapping['author'][target]]
            elif self.dataset=='IMDB':
                fidelity_out = nn.functional.softmax(self.model(fidelity_x_dict, self.computationalG_edge_index_dict), dim=-1)[mapping['movie'][target]]
                infidelity_out = nn.functional.softmax(self.model(infidelity_x_dict, self.computationalG_edge_index_dict), dim=-1)[mapping['movie'][target]]

        fidelity = self.orig_pred - fidelity_out[self.orig_pred_label].item()
        explanation_score = infidelity_out[self.orig_pred_label].item()
        infidelity = self.orig_pred - explanation_score

        if len(feature_exp)!=0:
            if self.dataset=='DBLP':
                feature_sparsity = {node:feature_exp[node].size / self.computationalG_x_dict[self.DBLP_idx2node_type[node%10]][mapping[self.DBLP_idx2node_type[node%10]][node//10]].count_nonzero().item() for node in feature_exp}
            elif self.dataset=='IMDB':
                feature_sparsity = {node:feature_exp[node].size / self.computationalG_x_dict['movie'][mapping['movie'][node]].count_nonzero().item() for node in feature_exp}
            overall_feat_sparsity = sum(feature_sparsity.values())/len(feature_sparsity.keys())
        else:
            feature_sparsity = {}
            overall_feat_sparsity = 0

        return fidelity, infidelity, explanation_score, (sum(feature_sparsity.values())+len(S)-len(feature_sparsity))/g.number_of_nodes(), len(S)/g.number_of_nodes(), overall_feat_sparsity, feature_sparsity

    def featureExpRaw2out(self, raw_feature_exp, mapping=None, target=None):
        
        feature_exp = {}

        if self.dataset=='DBLP':
            for node in raw_feature_exp:
                node_type = self.DBLP_idx2node_type[node%10]
                feature_exp[node] = self.computationalG_x_dict[node_type][
                    mapping[node_type][node//10]].nonzero().cpu().numpy().T[0][raw_feature_exp[node]]
        elif self.dataset=='IMDB':
            node_type = 'movie'
            for node in raw_feature_exp:
                feature_exp[node] = self.computationalG_x_dict[node_type][
                    mapping[node_type][node]].nonzero().cpu().numpy().T[0][raw_feature_exp[node]]
        elif self.dataset=='MUTAG':
            data = self.MUTAG_dataset[target]
            for node in raw_feature_exp:
                feature_exp[node] = data.x[node].nonzero().cpu().numpy().T[0][raw_feature_exp[node]]

        return  feature_exp

    def homoCalFidelity(self, target, S, feature_exp, evaluator=False):
        # S is the set of nodes in the explanation
        # feature_exp is a dict of feature explanation

        if len(S)==0:
            print('Empty explanation!')
            return

        self.orig_pred, self.orig_pred_label = self.getOrigPred(target)

        data = self.MUTAG_dataset[target]

        fidelity_x = copy.deepcopy(data.x)
        infidelity_x = torch.zeros_like(data.x)

        for node in S:
            if node in feature_exp:
                fidelity_x[node][feature_exp[node]] = 0
                infidelity_x[node][feature_exp[node]] = data.x[node][feature_exp[node]]
            else:
                continue

        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():

            data = self.MUTAG_dataset[target].to(self.device)
            fidelity_out = nn.functional.softmax(self.model(fidelity_x, self.MUTAG_dataset[target].edge_index, torch.tensor([0 for _ in range(fidelity_x.shape[0])]).to(self.device))[0], dim=-1) # model(x, edge_index, batch)
            infidelity_out = nn.functional.softmax(self.model(infidelity_x, self.MUTAG_dataset[target].edge_index, torch.tensor([0 for _ in range(infidelity_x.shape[0])]).to(self.device))[0], dim=-1) # model(x, edge_index, batch)

        fidelity = self.orig_pred - fidelity_out[self.orig_pred_label].item()
        explanation_score = infidelity_out[self.orig_pred_label].item()
        infidelity = self.orig_pred - explanation_score

        if len(feature_exp)!=0:
            ### this is wrong!
            # feature_sparsity = {node:feature_exp[node].size/self.MUTAG_dataset[target].x[node].count_nonzero().item() for node in feature_exp}
            ### this is wrong!
            feature_sparsity = {node:feature_exp[node].size/self.MUTAG_dataset[target].x[node].shape[0] for node in feature_exp}
            overall_feat_sparsity = sum(feature_sparsity.values())/len(feature_sparsity.keys())
        else:
            feature_sparsity = {}
            overall_feat_sparsity = 0

        return fidelity, infidelity, explanation_score, (sum(feature_sparsity.values())+len(S)-len(feature_sparsity))/self.MUTAG_dataset[target].x.shape[0], len(S)/self.MUTAG_dataset[target].x.shape[0], overall_feat_sparsity, feature_sparsity


    def readLabelDict(self):

        # root = osp.dirname(osp.realpath(__file__)).split('pytorch_geometric_hetero')[0]+'pytorch_geometric_hetero/'
        root = osp.join(osp.dirname(osp.realpath(__file__)))

        if self.dataset=='DBLP':
            pickle_path = root+'/data/'+self.dataset+'/'
            label_dict = {}
            label_dict['paper'] = pd.read_pickle(pickle_path+"papers_df.pkl").to_numpy()
            label_dict['author'] = pd.read_pickle(pickle_path+"authors_df.pkl").to_numpy()
            label_dict['term'] = pd.read_pickle(pickle_path+"terms_df.pkl").to_numpy()
            label_dict['conference'] = pd.read_pickle(pickle_path+"confs_df.pkl").to_numpy()

        return label_dict

    def printMeaningDBLP(self, S, feature_exp):

        root = osp.join(osp.dirname(osp.realpath(__file__)))

        ### check bag-of-words
        bag_of_words_paper = np.load(root+'/data/DBLP/bag_of_words_paper.npy', allow_pickle=True)
        bag_of_words_author = np.load(root+'/data/DBLP/bag_of_words_author.npy', allow_pickle=True)

        # pickle_path = 'data/DBLP/'
        # label_dict = {}
        # label_dict['paper'] = pd.read_pickle(pickle_path+"papers_df.pkl").to_numpy()
        # label_dict['author'] = pd.read_pickle(pickle_path+"authors_df.pkl").to_numpy()
        # label_dict['term'] = pd.read_pickle(pickle_path+"terms_df.pkl").to_numpy()
        # label_dict['conference'] = pd.read_pickle(pickle_path+"confs_df.pkl").to_numpy()
        # # print('raw drawn')
        # ###
        label_dict = self.readLabelDict()

        for node in S:
            if node%10==0:
                print(node)
                print(label_dict['author'][node//10][1:])
                if feature_exp[node] is not None:
                    try:
                        print(bag_of_words_author[feature_exp[node]])
                    except:
                        pass
            elif node %10==1:
                print(node)
                print(label_dict['paper'][node//10][1:])
                if feature_exp[node] is not None:
                    try:
                        print(bag_of_words_paper[feature_exp[node]])
                    except:
                        pass
            elif node%10==2:
                print(node)
                print(label_dict['term'][node//10][1:])
            else:
                print(node)
                print(label_dict['conference'][node//10][1:])

    def printMeaningIMDB(self, S, feature_exp=None):

        root = osp.join(osp.dirname(osp.realpath(__file__)))

        bag_of_words = np.load(root+'/data/IMDB/bag_of_words_movie.npy', allow_pickle=True)
        plot_keywords = np.load(root+'/data/IMDB/movie_plot_keywords.npy')
        with open(root+'/data/IMDB/movie_name.json') as f:
            movie_name = json.load(f)

        for node in S:
            print(node)
            print(movie_name[node])
            if feature_exp is not None:
                print(plot_keywords[node])
                print(bag_of_words[feature_exp[node]])

    def getSampleSize(self, target,  n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, p_threshold=0.05, pred_threshold=0.01):

        self.computationalG_x_dict = None
        self.computationalG_edge_index_dict = None

        if self.dataset=='DBLP':

            mapping, _, g = self.__subgraph__(target)
            # self.computationalG2device()
            num_RV = n_cat_value # for Phi_t

            for n in g.nodes:
                # print(n)
                node_type = self.DBLP_idx2node_type[n%10]
                node = mapping[node_type][n//10]
                num_RV += self.computationalG_x_dict[node_type][node].count_nonzero().item()
                # print(num_RV)
                
            num_samples = max(k*num_RV, num_samples)


        elif self.dataset=='IMDB':

            mapping, _, g = self.__subgraph__(target) 
            # self.computationalG2device()
            num_RV = n_cat_value

            node_type = 'movie'

            for n in g.nodes:
                # print(n)
                node = mapping[node_type][n]
                num_RV += self.computationalG_x_dict[node_type][node].count_nonzero().item()
                # print(num_RV)

            num_samples = max(k*num_RV, num_samples)
            # print(f'num_samples_needed is {num_samples_needed}')

        elif self.dataset=='MUTAG':

            data = self.MUTAG_dataset[target].to(self.device)

            num_samples = max(k*(data.x.count_nonzero().item()+n_cat_value), num_samples)
            # print(f'num_samples_needed is {num_samples_needed}')

        return num_samples

    def getNumRV(self, target,  n_cat_value=3, num_samples=1000, k=10, p_perturb=0.5, p_threshold=0.05, pred_threshold=0.01):

        self.computationalG_x_dict = None
        self.computationalG_edge_index_dict = None

        if self.dataset=='DBLP':

            mapping, _, g = self.__subgraph__(target)
            # self.computationalG2device()
            num_RV = 1 # for Phi_t

            for n in g.nodes:
                # print(n)
                node_type = self.DBLP_idx2node_type[n%10]
                node = mapping[node_type][n//10]
                num_RV += self.computationalG_x_dict[node_type][node].count_nonzero().item()
                # print(num_RV)
            
        elif self.dataset=='IMDB':

            mapping, _, g = self.__subgraph__(target) 
            # self.computationalG2device()
            num_RV = 1

            node_type = 'movie'

            for n in g.nodes:
                # print(n)
                node = mapping[node_type][n]
                num_RV += self.computationalG_x_dict[node_type][node].count_nonzero().item()
                # print(num_RV)

        elif self.dataset=='MUTAG':

            num_RV = self.MUTAG_dataset[target].x.count_nonzero().item()+1
            # print(f'num_samples_needed is {num_samples_needed}')

        return num_RV
