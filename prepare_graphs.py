import os
import math
import time
import argparse

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd

from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops, regionprops_table
from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

import torch
import torchvision.transforms as TF

from options.graph_options import GraphOptions
from ssn_pytorch.model import SSNModel
from ssn_pytorch.lib.ssn.ssn import sparse_ssn_iter
from data.image_operations import downsample_image
from data.cptac_slides import CPTAC_Slides

class GraphConstruction:
    def __init__(self, opt, model, dataset):
        self.opt = opt
        self.model = model
        self.dataset = dataset

    def run(self):

        for idx, slide in enumerate(self.dataset):

            # New blank slide for stitch and overlay
            s_w, s_h = slide.size
            stitch = downsample_image(slide.image, self.opt.downsample_factor, mode='numpy')[0]
            stitch = Image.fromarray(stitch, mode='RGB')

            node_features = torch.empty((slide.patch_count, 5))
            node_coords = torch.empty((slide.patch_count*self.opt.nspix, 2))

            for p_idx, patch_name in enumerate(slide.patches):

                patch_path = os.path.join(slide.tiles_path, patch_name)
                patch = np.array(Image.open(patch_path))

                # patch_name format: <slide_name>_<x>_<y>.png or <slide_name>_<col>_<row>.png
                patch_coords = patch_name.replace('.png','').split('_')[-2:] # get patch_coordinates
                p_h, p_w = patch.shape[0:2]
                x = int(patch_coords[0]) * int(p_w) # column
                y = int(patch_coords[1]) * int(p_h) # row

                spixel_labels, spixel_coords, spixel_features = self.predict_spixels(patch, x, y)

                # concatenate all spixels features to the slide level node features
                node_features = torch.cat((node_features, spixel_features), dim=0)

                # concatendate respective spixels centroids to slide level node coords
                node_coords = torch.cat((node_coords, spixel_coords))

                # stitch and overlay the superpixel segmentations
                x = int(patch_coords[0]) * int(p_w//self.opt.downsample_factor) # column
                y = int(patch_coords[1]) * int(p_h//self.opt.downsample_factor) # row

                spixel_labels = torch.from_numpy(spixel_labels).permute(2, 0, 1)
                spixel_labels = TF.functional.to_pil_image(spixel_labels)
                spixel_labels = spixel_labels.resize((p_w//self.opt.downsample_factor, p_h//self.opt.downsample_factor))

                # print(img.size)
                stitch.paste(im=spixel_labels, box=(x, y))

            stitch.save(os.path.join(slide.slide_path, slide.slide_name+'_stitch.png'))



    def predict_spixels(self, patch, x, y):

        h, w = patch.shape[:2]

        nspix_per_axis = int(math.sqrt(self.opt.nspix))
        self.opt.pos_scale = self.opt.pos_scale * max(nspix_per_axis/h, nspix_per_axis/w)

        coords = torch.stack(torch.meshgrid(torch.arange(h, device='cuda'), torch.arange(w, device='cuda')), 0)
        coords = coords[None].float()

        input = cv2.bilateralFilter(patch, 9, 75, 75)
        input = rgb2lab(input)
        
        input = torch.from_numpy(patch).permute(2, 0, 1)[None].to('cuda').float() # B, C, H, W

        input = torch.cat([self.opt.color_scale*input, self.opt.pos_scale*coords], 1)

        Q, H, feat = self.model(input)

        feat = feat.squeeze(0).permute(1,0).to('cpu').detach()
        labels = H.reshape(h, w).to('cpu').detach().numpy()

        if self.opt.enforce_connectivity:
            segment_size = h * w / self.opt.nspix
            min_size = int(0.1 * segment_size)
            max_size = int(5.0 * segment_size)
            visual_labels = _enforce_label_connectivity_cython(labels[None], min_size, max_size)[0]

        labels = np.where(labels == 0, labels.max()+1, labels)

        properties = regionprops_table(labels, properties=['label', 'centroid'])
        properties['label'][-1] = 0
        df = pd.DataFrame(properties).sort_values(by=['label'])

        spixel_centroids_y, spixel_centroids_x = df['centroid-0'], df['centroid-1']
        spixel_centroids_x = spixel_centroids_x + x
        spixel_centroids_y = spixel_centroids_y + y
        patch_spixel_coords = torch.tensor(list(zip(spixel_centroids_x, spixel_centroids_y)))

        patch_segments = mark_boundaries(patch, visual_labels)

        return patch_segments, patch_spixel_coords, feat







if __name__ == '__main__':

    opt = GraphOptions().parse()

    # dataloader for all the slides
    if 'CPTAC' in opt.dataroot:
        dataset = CPTAC_Slides(opt.dataroot, opt.slideroot, opt.slide_list)

    # loading the trained model
    if opt.weight is not None:
        model = SSNModel(opt.fdim, opt.nspix, opt.n_iter).to('cuda')
        model.load_state_dict(torch.load(opt.weight))
    else:
        model = lambda data: sparse_ssn_iter(data, opt.nspix, opt.n_iter)

    GraphConstruction(opt, model, dataset).run()

