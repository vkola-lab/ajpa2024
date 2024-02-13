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

class TcgaDataset(TissueDataset):
    def __init__(self, root, ids, fdim, n_classes, isTrain=False, transform=None):
        TissueDataset.__init__(self, root, ids, fdim, n_classes, isTrain)

        self.transform = transform

        self.to_be_predicted_classes = self.classdict
        # {'Normal': 0, 'TCGA-LUAD': 1, 'TCGA-LUSC': 2}

        # self.meta_feats = pd.read_csv(os.path.join('dataset/TCGA', 'clinical_metadata.csv')) # contains metadata for common samples.


    def fetch_label_from_code(self, label):

        # Label Conversion for 3-Label
        if label == 'TCGA-LUSC': label = 'lusc'
        elif label == 'TCGA-LUAD': label = 'luad'
        elif label == 'Normal': label = 'normal'

        if self.n_classes == 4:
            if label == 'normal':
                if 'TCGA-LUSC' in slide_name: label = 'lscc_normal'
                elif 'TCGA-LUAD' in slide_name: label = 'luad_normal'

        return label

    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        slide_name, label = info.split('\t')[0], info.split('\t')[1]
#        print(slide_name)

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        label = self.fetch_label_from_code(label)

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
