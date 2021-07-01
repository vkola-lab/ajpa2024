import os
import argparse
from itertools import combinations

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
from scipy.spatial import Delaunay

import torch
import torch.functional as tf


# def slic_lab(img_path, n_segments=50):
#     '''
#     Args:
#         img_path: path to the image
#         n_segments: approximate number of superpixels
#     Returns
#     '''
#     image = imageio.imread(img_path)
#     image2 = rgb2lab(image)
#     slic_output = slic(image, n_segments, enforce_connectivity=True)
#     regions = regionprops(slic_output)
#     superpixel_coord = []
#     for props in regions:
#         cy, cx = props.centroid
#         superpixel_coord.append((cy,cx))
#     slic_output = torch.tensor(slic_output)
#     num_classes = len(np.unique(slic_output.reshape(-1)))
#     one_hot = one_hot_embedding(slic_output, num_classes)
#     avgLAB = []
#     for idx_sp, matrix in enumerate(one_hot):
#         cache = zip(*np.nonzero(matrix)) #cache stores the position of 1's
#         sum = 0
#         for c in cache:
#             sum += image2[c[0]][c[1]]
#         num_of_ones = np.count_nonzero(matrix)
#         avgLAB.append(sum/num_of_ones)
#     return avgLAB

class BuildGraph(object):
    def __init__(self, root, slide, svs_root, downsample_factor=8, out_suffix=''):

        self.root = root # root folder containing all slide tiles
        self.slide = slide # slide name
        self.slide_path = os.path.join(self.root, self.slide)
        self.tiles_path = os.path.join(self.slide_path, self.slide+'_tiles')

        self.svs_root = svs_root

        self.downsample_factor = downsample_factor
        self.out_suffix = out_suffix

    def getStitchParameters(self, svs_root):

        svs_path = os.path.join(svs_root, self.slide+'.svs')
        svs = Image.open(svs_path)
        width, height = svs.size
        print('Dimensions of slide: ', width, height)

        return svs, (width, height)


    def __run__(self):
        svs, params = self.getStitchParameters(self.svs_root)
        width, height = params

        new_width, new_height = int(width // self.downsample_factor), int(height // self.downsample_factor)
        print(new_width, new_height)
        svs = svs.resize(size=(new_width, new_height), resample=Image.BILINEAR)

        stitch = Image.new('RGB', (width, height))

        self.patches = os.listdir(self.tiles_path)
        self.patch_count = len(self.patches)

        # for features & adj_s of the graph
        points = []
        graph_features = []

        for idx, patch_name in enumerate(self.patches):

            patch = Image.open(os.path.join(self.tiles_path, patch_name))

            # patch_name format: <slide_name>_<x>_<y>.png or <slide_name>_<col>_<row>.png
            coord = patch_name.replace('.png','').split('_')
            x = int(coord[-2]) * int(patch.size[0]) # column
            y = int(coord[-1]) * int(patch.size[0]) # row

            points.append([x, y])
            graph_features.append(torch.tensor([0,0,0], dtype=torch.float))
            # call slic_lab(patch) or call basic(patch, (x, y))

            # stitch patches
            stitch.paste(im=patch, box=(x, y))

        stitch = stitch.resize(size=(new_width, new_height), resample=Image.BILINEAR)
        overlay = Image.blend(svs, stitch, 0.5)
        overlay.save(os.path.join(self.slide_path, self.slide+self.out_suffix+'.png'))

        # for features and adj_s of the graph
        points = np.array(points)
        tri = Delaunay(points)

        edges = []
        for t in tri.simplices:
            for c in combinations(t, 2):
                edges.append(list(c))
        edges.extend([[i, j] for j, i in edges])
        edge_mat = torch.LongTensor(edges).transpose(0,1)
        graph_features = torch.stack(graph_features)

        torch.save(edge_mat, os.path.join(self.slide_path, 'adj_s.pt'))
        torch.save(graph_features, os.path.join(self.slide_path, 'features.pt'))




def main():
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--root', type=str, default='dataset/CPTAC', help='root directory of all preprocessed slides')
    parser.add_argument('--slide', required=True, type=str, default='', help='name of slide')
    parser.add_argument('--svs_root', type=str, default='/home/rushin/Documents/Research/HighResCNN/temp/SVS_Files/', help='root directory where all svs are stored')
    parser.add_argument('--out_suffix', type = str, default = '_overlay', help='output file suffix')

    args = parser.parse_args()

    so = BuildGraph(args.root, args.slide, args.svs_root, out_suffix=args.out_suffix).__run__()



if __name__ == '__main__':
    main()
