import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
)

import scipy.sparse as sp
import numpy as np

from itertools import product
from typing import Callable, List, Optional

from models import *

class DBLP(InMemoryDataset):

    # url = 'https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npy',
            'labels.npy', 'node_types.npy', 'train_val_test_idx.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass
        # path = download_url(self.url, self.raw_dir)
        # extract_zip(path, self.raw_dir)
        # os.remove(path)

    def process(self):
        data = HeteroData()

        node_types = ['author', 'paper', 'term', 'conference']
        for i, node_type in enumerate(node_types[:2]):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = torch.from_numpy(x.todense()).to(torch.float)

        x = np.load(osp.join(self.raw_dir, 'features_2.npy'))
        data['term'].x = torch.from_numpy(x).to(torch.float)

        node_type_idx = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
        data['conference'].num_nodes = int((node_type_idx == 3).sum())

        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['author'].y = torch.from_numpy(y).to(torch.long)

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = torch.from_numpy(idx).to(torch.long)
            mask = torch.zeros(data['author'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['author'][f'{name}_mask'] = mask

        s = {}
        N_a = data['author'].num_nodes
        N_p = data['paper'].num_nodes
        N_t = data['term'].num_nodes
        N_c = data['conference'].num_nodes
        s['author'] = (0, N_a)
        s['paper'] = (N_a, N_a + N_p)
        s['term'] = (N_a + N_p, N_a + N_p + N_t)
        s['conference'] = (N_a + N_p + N_t, N_a + N_p + N_t + N_c)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).to(torch.long)
                col = torch.from_numpy(A_sub.col).to(torch.long)
                data[src, dst].edge_index = torch.stack([row, col], dim=0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

class IMDB(InMemoryDataset):

    # url = 'https://www.dropbox.com/s/g0btk9ctr1es39x/IMDB_processed.zip?dl=1'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return [
            'adjM.npz', 'features_0.npz', 'features_1.npz', 'features_2.npz',
            'labels.npy', 'train_val_test_idx.npz'
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass
        # path = download_url(self.url, self.raw_dir)
        # extract_zip(path, self.raw_dir)
        # os.remove(path)

    def process(self):
        data = HeteroData()

        node_types = ['movie', 'director', 'actor']
        for i, node_type in enumerate(node_types):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = torch.from_numpy(x.todense()).to(torch.float)

        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['movie'].y = torch.from_numpy(y).to(torch.long)

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))
        for name in ['train', 'val', 'test']:
            idx = split[f'{name}_idx']
            idx = torch.from_numpy(idx).to(torch.long)
            mask = torch.zeros(data['movie'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['movie'][f'{name}_mask'] = mask

        s = {}
        N_m = data['movie'].num_nodes
        N_d = data['director'].num_nodes
        N_a = data['actor'].num_nodes
        s['movie'] = (0, N_m)
        s['director'] = (N_m, N_m + N_d)
        s['actor'] = (N_m + N_d, N_m + N_d + N_a)

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).to(torch.long)
                col = torch.from_numpy(A_sub.col).to(torch.long)
                data[src, dst].edge_index = torch.stack([row, col], dim=0)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

def loadDataset(dataset_name):

    if dataset_name.lower()=='dblp':
        dataset_name="DBLP"
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

        return x_dict, edge_index_dict, None, None

    elif dataset_name.lower()=='imdb':
        dataset_name = 'IMDB'
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/'+dataset_name)

        metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                     [('movie', 'director'), ('director', 'movie')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edges=True, drop_unconnected_nodes=True)
        dataset = IMDB(path, transform=transform)
        data = dataset[0]
        # print(data)

        return None, None, data, None

    elif dataset_name.lower()=='mutag':
        dataset_name = 'MUTAG'
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'TU')
        dataset = TUDataset(path, name=dataset_name)

        return None, None, None, dataset


def loadGNN(dataset_name):
    if dataset_name.lower()=='dblp':

        hidden_channels=64
        out_channels=4
        num_heads=2
        num_layers=2

        ckpt_name = '_'.join((dataset_name, 'inDim', str(hidden_channels), 'nHead', str(num_heads),'nLayer', str(num_layers)))
        ckpt_name+='_noTerm'
        ckpt_path = osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', ckpt_name+'.pt')

        node_types = {'author': 451, 'paper': 4233, 'conference': 1}
        meta = (['author', 'paper', 'conference'],
                [('author', 'to', 'paper'),
                 ('paper', 'to', 'author'),
                 ('paper', 'to', 'conference'),
                 ('conference', 'to', 'paper')])
        
        model = HGT(hidden_channels=hidden_channels, out_channels=out_channels, num_heads=num_heads, num_layers=num_layers, node_types=node_types, metadata = meta)

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['net'])

    if dataset_name.lower()=='imdb':
        dataset_name = 'IMDB'

        hidden_channels = 128
        out_channels = 3
        num_heads = 8
        metadata = (['movie'], [('movie', 'metapath_0', 'movie'), ('movie', 'metapath_1', 'movie')])

        ckpt_name = '_'.join((dataset_name, 'hDim', str(hidden_channels), 'nHead', str(num_heads)))
        ckpt_path = osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', ckpt_name+'.pt')

        model = HAN(in_channels=-1, out_channels=out_channels, hidden_channels=hidden_channels, heads=num_heads, metadata=metadata)

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['net'])
        print('Trained model loaded.')


    elif dataset_name.lower()=='mutag':
        dataset_name = 'MUTAg'

        ckpt_name = '_'.join((dataset_name, 'GCN'))
        ckpt_path = osp.join(osp.dirname(osp.realpath(__file__)), 'checkpoints', ckpt_name+'.pt')


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(7, 2).to(device)

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['net'])

    return model