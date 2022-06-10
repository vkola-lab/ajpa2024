import os
import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.utils.data as data

from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import dense_to_sparse

from sklearn.model_selection import StratifiedKFold

def read_file(file_name):
    with open(file_name, 'r') as f:
        records = list(f)

    return records

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

class TissueDataset(data.Dataset):
    def __init__(self, root, ids, fdim, c_type, isTrain=False):

        self.root = root
        self.ids = ids
        self.fdim = fdim
        self.c_type = c_type
        self.isTrain = isTrain

        if self.c_type == '3_WAY':
            self.classdict = {'normal': 0, 'lusc': 1, 'luad': 2}
        elif self.c_type == '4_WAY':
            self.classdict = {'lscc_normal': 0, 'luad_normal': 1, 'lscc_tumor': 2, 'luad_tumor': 3}
        else:
            raise ValueError("Invalid classification type.")

        self.unique_classes = set(self.classdict.values())
        self.num_classes = len(self.unique_classes)

        self.eval_metrics = ['acc', 'precision', 'recall', 'cm'] # measure accuracy of the data

        self._up_kwargs = {'mode': 'bilinear'}

    def get_slide_attributes(self, slide_name):
        """Slide attributes fetch, common for all datasets
        Parameters:
            slide_name (String): The name of the slide itself representing the path of the slide.
        """
        # Loading Node Features
        feature_path = os.path.join(self.root, slide_name, 'features.pt')
        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            print(feature_path + ' not exists')
            features = torch.zeros(1, self.fdim)

        # Loading Node Edge Indices
        adj_s_path = os.path.join(self.root, slide_name, 'adj_s_ei.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)
        else: # never going to be triggered
            print(adj_s_path + ' not exists')
            adj_s = torch.ones(features.shape[0], features.shape[0])

        # # Loading Edge Attributes
        # edge_attr_path = os.path.join(self.root, slide_name, 'edge_attr.pt')
        # if os.path.exists(edge_attr_path):
        #     edge_attr = torch.load(edge_attr_path, map_location=lambda storage, loc: storage)
        # else: # never going to be triggered
        #     print(edge_attr_path + ' not exists')
        #     edge_attr = torch.ones(features.shape[0], features.shape[0])

        # Loading Node Coordinates
        if not self.isTrain:
            node_coord_path = os.path.join(self.root, slide_name, 'c_idx.txt')
            if os.path.exists(node_coord_path):
                node_coords = []
                c_idx = read_file(node_coord_path)
                for node in c_idx:
                    node = node.replace('\n', '')
                    x, y = node.split('\t')[0], node.split('\t')[1]
                    node_coords.append((x,y))
            else: # never going to be triggered
                print(node_coord_path + ' not exists')
                node_coords = [(0,0)]*features.shape[0]
        elif self.isTrain:
            node_coords = None

        # return features, adj_s, edge_attr, node_coords
        return features, adj_s, node_coords


    @abstractmethod
    def __getitem__(self, index): # maybe call this function from the inheriting function for common functionality?
        """Unpack and create a graph data object. Specific to each dataset and needs to implemented in the child class
        Parameters:
            index (int): Index of the slide from a slide_list given by 'ids'
        """

        pass

    def __len__(self):
        return len(self.ids)
