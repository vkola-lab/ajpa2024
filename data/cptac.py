import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

from sklearn.model_selection import StratifiedKFold

def separate_data(graph_list, seed, n_folds, fold_idx):
    assert 0 <= fold_idx and fold_idx < n_folds, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=n_folds, shuffle = True, random_state = seed)

    labels = []
    for info in graph_list:
        info = info.replace('\n', '')
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], info.split('\t')[1]
        labels.append(label)
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, val_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    val_graph_list = [graph_list[i] for i in val_idx]

    return train_graph_list, val_graph_list


class S2VGraph(object):
    def __init__(self, id, node_features, edge_mat, label):
        '''
            id: The file name of each graph i.e WSI name
            label: an integer graph label
            node_features: a torch float tensor, representing each node as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
        '''
        self.id = id
        self.label = label
        self.node_features = node_features
        self.edge_mat = edge_mat

class CPTAC_Nodes(data.Dataset):
    def __init__(self, root, ids, fdim):
        super(CPTAC_Nodes, self).__init__()

        self.root = root
        self.ids = ids
        self.fdim = fdim
        self.classdict = {'normal': 0, 'luad': 1, 'lscc': 2}
        self._up_kwargs = {'mode': 'bilinear'}

    def __getitem__(self, index):
        info = self.ids[index].replace('\n', '')
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], info.split('\t')[1]

        feature_path = os.path.join(self.root, file_name, 'features.pt')
        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            print(feature_path + ' not exists')
            features = torch.zeros(1, self.fdim)

        adj_s_path = os.path.join(self.root, file_name, 'adj_s.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)
        else:
            print(adj_s_path + ' not exists')
            adj_s = torch.ones(features.shape[0], features.shape[0])

        # features = features.unsqueeze(0)
        sample = S2VGraph(id=file_name, node_features=features, edge_mat=adj_s, label=self.classdict[label])

        return sample

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':

    def read_file(file_name):
        with open(file_name, 'r') as f:
            records = list(f)

        return records

    train_file = '/home/rushin/Documents/Research/HighResCNN/temp/CPTAC/train_wsi.txt'
    train_ids = read_file(train_file)

    root = '/home/rushin/Documents/Research/HighResCNN/temp/CPTAC'
    train_graphs = CPTAC_Nodes(root, train_ids)

    print(train_graphs.classdict)

    graph_obj = train_graphs[0]
    print(graph_obj.id, graph_obj.label, graph_obj.node_features.shape, graph_obj.edge_mat.shape)
