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

    print("Num of centroid coordinates: ", len(superpixel_coords))
    print("Num of spixel_features: ", len(avgLAB))
    print("Shape of spixel_features: ", avgLAB[0])

    return patch_overlay, superpixel_coords, avgLAB


if __name__ == '__main__':

    tiles_path = '/home/rushin/Documents/Research/HighResCNN/graphCNN/dataset/CPTAC/C3L-05022-21/C3L-05022-21_tiles'
    patch_name = 'C3L-05022-21_7_3.png'

    patch = Image.open(os.path.join(tiles_path, patch_name))
    so = slic_lab(np.array(patch), n_segments=32)
