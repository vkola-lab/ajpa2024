import random
import numpy as np
import torch.utils.data as data

class GCNGraph(data.dataset):
    ''' Input Graph and Label '''

    def __init__(self, root, ids, target_patch_size=-1):
        super GCNGraph, self).__init__()
        '''
        Args:

        root:
        ids:
        target_patch_size:
        '''

        self.root = root
        self.ids = ids
        # self.target_patch_size = target_patch_size
        self.classdict = {'normal': 0, 'luad': 1, 'lscc': 2}
        # self.classdict = {'Normal': 0, 'TCGA-LUAD': 1, 'TCGA-LUSC': 2}
        self._up_kwargs = {'mode': 'bilinear'}

    def __getitem__(self, index):
        sample = {}
        info = self.ids[index].replace('\n', '')
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], infor.split('\t')[1]
        sample['label'] = self.classdict[label]
        sample['id'] = file_name

        feature_path = os.path.join(self.root, file_name, 'features.pt')
        if os.path.exists(feature_path):
            features = torch.load(feature_path, map_location=lambda storage, loc: storage)
        else:
            print(feature_path + ' not exists')
            features = torch.zeros(1, 512)

        adj_s_path = os.path.join(self.root, file_name, 'adj_s.pt')
        if os.path.exists(adj_s_path):
            adj_s = torch.load(adj_s_path, map_location=lambda storage, loc: storage)

        else:
            print(adj_s_pat + ' not exists')
            adj_s = torch.ones(features.shape[0], features.shape[0])

        # features = features.unsqueeze(0)
        sample['image'] = features
        sample['adj_s'] = adj_s # adj_s.to(torch.double)
        # return {'image': image.astype(np.float32), 'label': label.astype(np.int64)}

        return sample
