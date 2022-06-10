import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import dense_to_sparse

from sklearn.model_selection import StratifiedKFold

from .base_dataset import TissueDataset


class CptacDataset(TissueDataset):
    def __init__(self, root, ids, fdim, c_type, isTrain=False):
        TissueDataset.__init__(self, root, ids, fdim, c_type, isTrain)

        self.to_be_predicted_classes = self.classdict

    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], info.split('\t')[1]
        slide_name = file_name.split('/')[1]

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        # Label Conversion for 3-Label / 4-Label classification
        if label == 'lscc': label = 'lusc'
        elif label == 'luad': label = 'luad'

        if self.c_type == '4_WAY':
            if label == 'normal':
                if 'LSCC' in file_name: label = 'lscc_normal'
                elif 'LUAD' in file_name: label = 'luad_normal'

        # Custom Data Object with slide_name & node_coordinates
        geometric_graph = Data(x=features,
                               edge_index=adj_s,
                               edge_attr=None,
                               y=self.classdict[label],
                               slide_path=file_name,
                               node_coords=node_coords)

        return geometric_graph

    def __len__(self):
        return len(self.ids)
