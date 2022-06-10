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


class PcgaDataset(TissueDataset):
    def __init__(self, root, ids, fdim, c_type, isTrain=False):
        TissueDataset.__init__(self, root, ids, fdim, c_type, isTrain)

        # self.classdict = {'pml_normal': 0, 'hyperplasia': 1, 'metaplasia': 2, 'mild_dysplasia': 3, 'moderate_dysplasia': 4, 'severe_dysplasia': 5, 'cis': 6, 'unknown': 7, 'tumor': 8}
        self.classdict = {'pml_normal': 0, 'hyperplasia':1, 'metaplasia': 2, 'high_grade': 3, 'unknown': 4}
        # self.classdict = {'premalignant': 0}
        if self.c_type == '3_WAY':
            self.to_be_predicted_classes = {'normal': 0, 'lusc': 1, 'luad': 2}
        elif self.c_type == '4_WAY':
            self.to_be_predicted_classes = {'lscc_normal': 0, 'luad_normal': 1, 'lscc_tumor': 2, 'luad_tumor': 3}
        else:
            raise ValueError("Invalid classification type.")

        self.meta_feats = pd.read_csv(os.path.join('dataset/PCGA/', 'clinical_metadata.csv')) # Save a corresponding pcga_rna.txt file

    def fetch_label_from_code(self, label):

        if label == 'NA' or label == 'N/A':
            label = 'unknown'
        elif float(label) < 22:
            label = 'pml_normal'
        elif float(label) >= 22 and float(label) < 23:
            label= 'hyperplasia'
        elif float(label) >= 23 and float(label) < 24:
            label = 'metaplasia'
        elif float(label) >= 24 and float(label) < 25:
            # label = 'dysplasia' # 'mild_dysplasia'
            label = 'high_grade'
        elif float(label) >= 25 and float(label) < 26:
            # label = 'dysplasia' # 'moderate_dysplasia'
            label = 'high_grade'
        elif float(label) >= 26 and float(label) < 27:
            # label = 'dysplasia' # 'severe_dysplasia'
            label = 'high_grade'
        elif float(label) >= 27 and float(label) < 28:
            # label = 'cis'
            label = 'high_grade'
        elif float(label) > 28:
            # label = 'tumor'
            label = 'high_grade'
        return label


    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        file_name, label = info.split('\t')[0], info.split('\t')[1] # .rsplit('.', 1)[0]
        slide_name = file_name # .split('/')[1]

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        # regarding labels:
        label = self.fetch_label_from_code(label)
        # label = "premalignant"

        geometric_graph = Data(x=features,
                               edge_index=adj_s,
                               edge_attr=None,
                               y=self.classdict[label],
                               slide_path=file_name,
                               node_coords=node_coords)

        return geometric_graph

    def __len__(self):
        return len(self.ids)
