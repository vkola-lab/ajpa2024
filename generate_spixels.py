import os
import argparse
from itertools import combinations

import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 933120000
from scipy.spatial import Delaunay

import torch
import torch.nn.functional as tnf

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from skimage.color import rgb2lab

from util import read_file

def slic_lab(image, n_segments=50, x_offset=0, y_offset=0):
    '''
    Args:
        img_path: path to the image
        n_segments: approximate number of superpixels
    Returns
    '''
    image_lab = torch.from_numpy(rgb2lab(image))
    # print(image_lab.shape)

    slic_output = slic(image, n_segments, enforce_connectivity=True, start_label=1)

    regions = regionprops(slic_output)
    superpixel_coords = []
    for props in regions:
        cy, cx = props.centroid
        superpixel_coords.append((cx + x_offset, cy + y_offset))

    # print("Num of centroid coordinates: ", len(superpixel_coords))

    patch_overlay = mark_boundaries(image, slic_output, (0,0,0), (0,0,0))
    patch_overlay = Image.fromarray((patch_overlay * 255).astype(np.uint8))
    # q.save('/home/rushin/Documents/Research/HighResCNN/graphCNN/dataset/CPTAC/C3L-05022-21/C3L-05022-21_4_11_spixel.png')

    slic_output = torch.from_numpy(slic_output)

    if 0 not in slic_output:
        slic_output = slic_output - 1

        num_classes = len(torch.unique(slic_output))
        # print("Num of classes: ", num_classes)

        one_hot = tnf.one_hot(slic_output, num_classes).permute(2, 0, 1)
        # print(one_hot.shape)

    elif 0 in slic_output:
        num_classes = len(torch.unique(slic_output))
        # print("Num of classes: ", num_classes)

        one_hot = tnf.one_hot(slic_output, num_classes).permute(2, 0, 1)
        # print(one_hot.shape)
        one_hot = one_hot[1:,:,:]
        # print(one_hot.shape)


    avgLAB = []
    for idx_sp, matrix in enumerate(one_hot):
        idx = matrix.nonzero(as_tuple=True) # cache stores the position of 1's
        # print("Image Lab shape", image_lab[idx[0], idx[1], :].shape)
        sum = torch.sum(image_lab[idx[0], idx[1], :], dim=0)
        # print("Sum shape: ", sum.shape)
        num_of_ones = len(idx[0])
        avgLAB.append((sum/num_of_ones).type(torch.float))

    # print("Num of centroid coordinates: ", len(superpixel_coords))
    # print("Num of spixel_features: ", len(avgLAB))
    # print("Shape of spixel_features: ", avgLAB[0])

    return patch_overlay, superpixel_coords, avgLAB

class BuildGraph(object):
    def __init__(self, root, slide, svs_root, downsample_factor=8, out_suffix='', n_segments=32):

        self.root = root # root folder containing all slide tiles
        self.slide = slide # slide name
        self.slide_path = os.path.join(self.root, self.slide)
        self.tiles_path = os.path.join(self.slide_path, self.slide+'_tiles')

        self.svs_root = svs_root

        self.downsample_factor = downsample_factor
        self.out_suffix = out_suffix

        self.n_segments = n_segments

        print(self.slide)

    def getStitchParameters(self, svs_root):

        svs_path = os.path.join(svs_root, self.slide+'.svs')
        svs = Image.open(svs_path)
        width, height = svs.size
        print('Dimensions of slide: ', width, height)

        return svs, (width, height)


    def __run__(self):

        # Stitching parameters
        # svs, params = self.getStitchParameters(self.svs_root)
        # width, height = params
        #
        # new_width, new_height = int(width // self.downsample_factor), int(height // self.downsample_factor)
        # print(new_width, new_height)
        # svs = svs.resize(size=(new_width, new_height), resample=Image.BILINEAR)
        #
        # stitch = Image.new('RGB', (width, height))


        # features & edge_mat
        self.patches = os.listdir(self.tiles_path)
        self.patch_count = len(self.patches)
        print(self.patch_count)

        # for features & adj_s of the graph
        points = []
        graph_features = []

        for idx, patch_name in enumerate(self.patches):

            patch = Image.open(os.path.join(self.tiles_path, patch_name))

            # patch_name format: <slide_name>_<x>_<y>.png or <slide_name>_<col>_<row>.png
            coord = patch_name.replace('.png','').split('_')
            x = int(coord[-2]) * int(patch.size[0]) # column
            y = int(coord[-1]) * int(patch.size[0]) # row

            patch_overlay, superpixel_coords, superpixel_features = slic_lab(np.array(patch), self.n_segments, x_offset=x, y_offset=y)

            points.extend(superpixel_coords)
            graph_features.extend(superpixel_features)

            # stitch patches
            # stitch.paste(im=patch_overlay, box=(x, y))


        # stitch = stitch.resize(size=(new_width, new_height), resample=Image.BILINEAR)
        # overlay = Image.blend(svs, stitch, 0.5)
        # overlay.save(os.path.join(self.slide_path, self.slide+self.out_suffix+'.png'))

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
    parser.add_argument('--preprocess_file', required=True, type=str, default='train_wsi.txt', help='File with list of slides to preprocess')
    parser.add_argument('--svs_root', type=str, default='/home/rushin/Documents/Research/HighResCNN/temp/SVS_Files/', help='root directory where all svs are stored')
    parser.add_argument('--out_suffix', type = str, default = '_overlay', help='output file suffix')
    parser.add_argument('--n_segments', type = str, default = 32, help='Approx frequency of superpixels per patch')

    args = parser.parse_args()

    file = os.path.join(args.root, args.preprocess_file)
    slide_list = read_file(file)

    for info in slide_list:
        info = info.replace('\n', '')
        file_name, label = info.split('\t')[0].rsplit('.', 1)[0], info.split('\t')[1]
        so = BuildGraph(args.root, file_name, args.svs_root, downsample_factor=4, out_suffix=args.out_suffix, n_segments=args.n_segments).__run__()



if __name__ == '__main__':
    main()
