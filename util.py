import os
import os.path as osp
from itertools import product
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

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