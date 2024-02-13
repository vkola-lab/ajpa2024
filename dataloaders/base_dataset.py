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

def read_file(file_name):
    with open(file_name, 'r') as f:
        records = list(f)

    return records

class TissueDataset(data.Dataset):
    def __init__(self, root, ids, fdim, n_classes, isTrain=False):

        self.root = root # path for simclr_files
        self.ids = ids # wsi_ids to use in dataloader
        self.fdim = fdim # feature dimension of each node
        self.n_classes = n_classes # Is it 3 way classification or 4 way classification?
        self.isTrain = isTrain # is it training or evaluation?

        if self.n_classes == 3:
            self.classdict = {'normal': 0, 
                              'lusc': 1, 
                              'luad': 2}

        elif self.n_classes == 4: # never used most_likely
            self.classdict = {'lscc_normal': 0, 
                              'luad_normal': 1, 
                              'lscc_tumor': 2, 
                              'luad_tumor': 3}
        else:
            raise ValueError("Invalid classification type.")

        self.unique_classes = set(self.classdict.values())
        self.num_classes = len(self.unique_classes)

        self.eval_metrics = ['acc', 'precision', 'recall', 'cm', 'rocauc', 'F1'] # measure accuracy of the data

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
        
        node_coord_path = os.path.join(self.root, slide_name, 'c_idx.txt')
        if os.path.exists(node_coord_path):
            node_coords = []
            c_idx = read_file(node_coord_path)
            for node in c_idx:
                node = node.replace('\n', '')
                x, y = node.split('\t')[0], node.split('\t')[1]
                node_coords.append((int(x),int(y)))
            node_coords = torch.tensor(node_coords)
        else: # never going to be triggered
            print(node_coord_path + ' not exists')
            node_coords = [(0,0)]*features.shape[0]
            node_coords = torch.tensor(node_coords)
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
