import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

from torch_geometric.data import Data, Batch, DataLoader

from sklearn.model_selection import StratifiedKFold

from .base_dataset import TissueDataset


class CptacDataset(TissueDataset):
    def __init__(self, root, ids, fdim, n_classes, isTrain=False, transform=None):
        TissueDataset.__init__(self, root, ids, fdim, n_classes, isTrain)

        self.transform = transform

        self.to_be_predicted_classes = self.classdict

    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        slide_name, label = info.split('\t')[0], info.split('\t')[1]

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        # Label Conversion for 3-Label / 4-Label classification
        if self.n_classes == 3:
            if label == 'lscc': label = 'lusc'
            elif label == 'luad': label = 'luad'

        elif self.n_classes == 4:
            if label == 'normal':
                if 'LSCC' in slide_name: label = 'lscc_normal'
                elif 'LUAD' in slide_name: label = 'luad_normal'

        # Custom Data Object with slide_name & node_coordinates
        geometric_graph = Data(x=features,
                               edge_index=adj_s,
                               edge_attr=None,
                               y=self.classdict[label],
                               slide_path=slide_name,
                               node_coords=node_coords)

        if self.transform:
            geometric_graph = self.transform(geometric_graph)

        return geometric_graph

    def __len__(self):
        return len(self.ids)
