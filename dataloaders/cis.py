import os
import time
import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.utils.data as data

from torch_geometric.data import Data, Batch, DataLoader

from sklearn.model_selection import StratifiedKFold

from .base_dataset import TissueDataset


class CisDataset(TissueDataset):
    def __init__(self, root, ids, fdim, n_classes, isTrain=False, transform=None):
        TissueDataset.__init__(self, root, ids, fdim, n_classes, isTrain=False)

        self.transform = transform

        # self.classdict = {'pml_normal': 0, 'hyperplasia': 1, 'metaplasia': 2, 'mild_dysplasia': 3, 'moderate_dysplasia': 4, 'severe_dysplasia': 5, 'cis': 6, 'unknown': 7, 'tumor': 8}
        self.classdict = {'cis': 0}
        if self.n_classes == 3:
            self.to_be_predicted_classes = {'normal': 0, 
                                            'lusc': 1,
                                             'luad': 2}

        elif self.n_classes == 4:
            self.to_be_predicted_classes = {'lscc_normal': 0, 
                                            'luad_normal': 1, 
                                            'lscc_tumor': 2, 
                                            'luad_tumor': 3}
        else:
            raise ValueError("Invalid classification type.")

        # self.meta_feats = pd.read_csv(os.path.join('dataset/CIS', 'clinical_metadata.csv')) # contains metadata for common samples.

    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        slide_name, label = info.split('\t')[0], info.split('\t')[1] 

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        if label == 'CIS': 
            label = 'cis'

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
