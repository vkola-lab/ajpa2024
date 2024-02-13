import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math 
import os
import numpy as np
import cv2
import json
import csv

import openslide
from openslide import open_slide, ImageSlide, OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000

import torch

import numpy as np
import matplotlib.cm as mpl_color_map
import matplotlib.colors as mpl_colors
from scipy.stats import zscore

def show_cam_on_image(img, mask, tissue_map):
    print("Img and Mask shapes: ", img.shape, mask.shape)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap * tissue_map[..., None]
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    return cam


# LR Propogation from Transformer-Explainability
# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def generate_relevance(model, input, index=None):
    output = model(input, register_hook=True)
    prob = torch.nn.functional.softmax(output, dim=1)
    y_pred = torch.argmax(output.detach(), dim=1).view(-1, 1).cpu()
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    prob = prob[0, index]
    # print("Probability LUSC :", prob)
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    # get relevance maps of self-attn encoders
    
    # for idx, (pma, encoders) in enumerate(model.pool.layers):
        
    pma, encoders = model.pool.layers[-1]    
    num_tokens = encoders[0].mab.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    
    for encoder in encoders:
        grad = encoder.mab.get_attn_gradients()
        cam = encoder.mab.get_attention_map()
        cam = avg_heads(cam, grad)

        R += apply_self_attention_rules(R.cuda(), cam.cuda()) # (num_tokens X num_tokens)

    # R = torch.eye(assignment_shape[0], assignment_shape[1]).cuda()
    grad_pma = pma.mab.get_attn_gradients()
    cam_pma = pma.mab.get_attention_map()

    cam_pma = avg_heads(cam_pma, grad_pma) # (num_tokens-1 X num_nodes) 

    # R --> num_tokens x num_tokens
    # R --> 1 x num_tokens - 1
    R_pma = torch.matmul(R[0:1, 1:].cuda(), cam_pma.cuda()) # R_pma --> 1 x num_nodes
        # break

    return R_pma[0, :].detach().cpu(), output.detach().cpu(), y_pred

def NormalizeData(data):
    return (data - data.min()) / (data.max() - data.min())

def fetch_slide_image(slide_path, slide_root, patch_size=256, overlap=1, downsample_factor=16.0, gt='na', save_path=None):
    
    slide_name = slide_path
    if 'CIS' in slide_root: ext = 'ndpi'
    else: ext = 'svs' 

    if os.path.isfile(os.path.join(slide_root, '{}.{}'.format(slide_path, ext))):
        slide_path = os.path.join(slide_root, '{}.{}'.format(slide_path, ext))
    else:
        print(os.path.join(slide_root, '{}.{}'.format(slide_path, ext)))
        slide_path = os.path.join(slide_root, '{}.{}'.format(slide_path, 'tif'))

    slide = open_slide(slide_path)
    print(slide.dimensions)
    try:
        slide_magnification = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    except:
        slide_magnification = '20'

    if slide_magnification == '40': # Downsample first to 20x. Then use img_20x to downsample further

        print("Magnification: 40x")
        Objective = int(slide_magnification)
        Factors = slide.level_downsamples
        Available = tuple(Objective / x for x in Factors)

        if len(Available) > 1:
            level = Available.index(20.0)
            img_20x = slide.read_region((0,0), level, slide.level_dimensions[level])  
        else:  
            dz = DeepZoomGenerator(slide, patch_size, overlap=1, limit_bounds=False)

            for level in range(dz.level_count-1, -1, -1):
                level_mag = Available[0]/pow(2, dz.level_count-(level+1))

                if level_mag == 20.0:
                    # Get the size of the downsampled image
                    w, h = dz.level_dimensions[level]

                    # Create an empty PIL image with the correct size
                    img_20x = slide.get_thumbnail((w,h))
                    break
                    
        slide_20x = ImageSlide(img_20x)
        
    else:
        print("Magnification: 20x")
        slide_20x = slide

    dz_20x = DeepZoomGenerator(slide_20x, patch_size, overlap=1, limit_bounds=False)
    level = dz_20x.level_count - 1 - int(math.log(downsample_factor, 2))
    # Get the size of the downsampled image
    w, h = dz_20x.level_dimensions[level]
    # Create an empty PIL image with the correct size
    
    downsampled_img = slide_20x.get_thumbnail((w,h))
    if save_path:
        if not os.path.exists(os.path.join(save_path, slide_name+"_true_{}.png".format(gt.item()))):
            downsampled_img.save(os.path.join(save_path, slide_name+"_true_{}.png".format(gt.item())))

    return downsampled_img

def plot_heat_maps(graph, scores, slide_root, patch_size=256, overlay=True, clamp=0.05, norm=True, colormap='RdBu_r', save_path=None):
    
    # Fetch patch coords & slide path for the tissue
    slide_path = graph.slide_path[0]
    coords = graph.node_coords
    # coords = [(int(x), int(y)) for x,y in coords]

    # fetch tissue image at specific downsample
    downsample_factor = 16.0
    image = fetch_slide_image(slide_path, slide_root, patch_size, downsample_factor=downsample_factor, gt = graph.y, save_path=save_path)
    image = np.asarray(image.convert("RGB"))
    image = (image - image.min()) / (image.max() - image.min())

    y_min, y_max, x_min, x_max = 0, image.shape[0], 0, image.shape[1]

    attention_map = np.zeros((image.shape[0], image.shape[1]), dtype=bool) # this is the cam mask   
    tissue_map = -np.ones((image.shape[0], image.shape[1]), dtype=np.float32) # this is the image
    
    offset = patch_size + 2 # 2 is for overlap
    d = downsample_factor
    scores = scores.numpy()
    scores = zscore(scores)

    if clamp:
        q05, q95 = torch.quantile(scores, clamp), torch.quantile(scores, 1-clamp)
        scores = np.clip(scores, a_min=q05, a_max=q95)

    # check if all values in scores are 0s
    scores = np.nan_to_num(scores, nan=0)
    if not np.all(scores == -1):
        scores = MinMaxScaler(feature_range=(-1, 1)).fit_transform(scores.reshape(-1,1))

    for (x,y), s in zip(coords, scores):

        x, y = x*patch_size, y*patch_size
        
        mask[round(y.item()/d):round((y.item()+offset)/d), round(x.item()/d):round((x.item()+offset)/d)] = True
        heatmap[round(y.item()/d):round((y.item()+offset)/d), round(x.item()/d):round((x.item()+offset)/d)] = s
            
    plt.figure(figsize=(30, 30))
    a = 1.
    if overlay:
        plt.imshow(image, alpha=1, cmap='gray')
        a = 0.7

    plt.imshow(heatmap, alpha=0.5*mask, cmap=colormap, interpolation='nearest')
    cbar = plt.colorbar(location='right', orientation='vertical')
    cbar.ax.tick_params(labelsize=40)
    plt.axis('off')

    return plt

def binaryMaskIOU(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and(mask1==1,  mask2==1))
    iou = intersection/(mask1_area+mask2_area-intersection)

    return iou

def plot_heatmaps_iou(graph, scores, slide_gt, slide_root, dataset, patch_size=256, overlay=True, clamp=0.05, norm=True, colormap='jet', crop=False):
    doub = ['C3L-04759-25', 'C3L-02654-21']
    doub_left = ['C3L-02625-22']
    print(dataset)
    if dataset =='cptac':
        if slide_gt == 2:
            tumor_colors = [
                [255, 196, 160],
                [255,0,0],
                [42,0,128],
                [198,233,255],
                [114,188,178],
                [182,0,255],
                [60,128,0],
                [0,243,255],
                [180,162,255]
            ]
        else:
            tumor_colors = [
                [255, 196, 160],
                [0, 255, 0],
                [255,0,0],
            ]      
    else:
        tumor_colors = [
                [0,255,0],
                [255,0,158],
                [130,0,43],
                [60,128,0],
                [0,255,170],
                [0,255,255],
                [105,105,105],
                [255,0,0],
                [150,150,0],
                [255,0,255],
                [36,0,255],
                [255,158,0],
                [182,0,255],
                [255,202,202],
                [198,233,255],
                [197,132,255],
                [255,218,0],
                [105,134,255],
                [0,243,255],
                [114,188,178],
                [255,196,160],
                [0,100,150],
                [255,196,160],
                [255,182,63],
                [180,162,255],
                [152,0,0],
                [14,0,150],
                [255,103,103],
                [0,146,255],
                [255,0,0],
                [130,0,43],
                [42,0,128],
                [255,0,158],
                [255,158,0],
                [36,0,255],
                [0,255,0],
                [60,128,0],
                [182,0,255],
                [198,233,255],
                [0,243,255],
                [114,188,178],
                [180,162,255],
            ]   
    if dataset == 'pcga':
        f = open('../datasets/PCGA/measurements.json')
        pcga_crop = json.load(f)
        if graph.slide_path[0] in pcga_crop:
            Cx, Cy = float(pcga_crop[graph.slide_path[0]]['Cx']), float(pcga_crop[graph.slide_path[0]]['Cy'])
            A = float(pcga_crop[graph.slide_path[0]]['A'])
            P = float(pcga_crop[graph.slide_path[0]]['P'])

            # Solve for width and height
            w = (P/2 + math.sqrt((P/2)**2 - 4*A)) / 2
            h = A / w

            # Calculate top-left corner coordinates
            x1 = Cx - w/2
            y1 = Cy - h/2

            # # Print results
            # print(f"Width: {w:.2f}")
            # print(f"Height: {h:.2f}")
            # print(f"Top-left corner: ({x1:.2f}, {y1:.2f})")
            w, h = int(w), int(h)
            x1, y1 = int(x1), int(y1)
            crop = True

    # Fetch patch coords & slide path for the tissue
    slide_path = graph.slide_path[0]
    coords = graph.node_coords[0]
    coords = [(int(x), int(y)) for x,y in coords]
    
    # fetch tissue image at specific downsample
    downsample_factor = 16.0
    image = fetch_slide_image(slide_path, slide_root, patch_size, downsample_factor=downsample_factor)
    image = np.asarray(image.convert("RGB"))
    image = (image - image.min()) / (image.max() - image.min())

    y_min, y_max, x_min, x_max = 0, image.shape[0], 0, image.shape[1]

    annotation = Image.open('/SeaExp/Rushin/datasets/{}/Annotations/{}-annotations.png'.format(dataset.upper(), slide_path)).convert('RGB')
    # if crop:
    #     # w_factor = width//w
    #     # h_factor = height//hs
    #     # print(x1, y1, x1+w, y1+h)
    #     #annotation = annotation.crop((x1*2, y1*2, (x1+w)*2, (y1+h)*2))
    #     #annotation = annotation.crop((y1*2,x1*2, (y1+h)*2,(x1+w)*2))
    annotation = annotation.resize((image.shape[1], image.shape[0]))
    annotation = np.array(annotation)

    # generate annotation
    output_gt = np.zeros((annotation.shape[0],annotation.shape[1],3), np.uint8)
    output_gt[:] = (255,255,255)
    gt = np.ones((image.shape[0],image.shape[1]))
    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            pixel = annotation[i][j]
            r, g, b = annotation[i][j]
            for r_th, g_th, b_th in tumor_colors:
                if r<r_th+10 and r>r_th-10 and g<g_th+10 and g>g_th-10 and b<b_th+10 and b>b_th-10:
                    output_gt[i][j] = (0,0,0)
                    gt[i][j] = 0

    attention_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)    
    tissue_map = -np.ones((image.shape[0], image.shape[1]), dtype=np.float32)
    
    offset = patch_size+2 # need to set according to the magnification of scanning vs extraction i.e if scanned 40x but extracted 20x (512x512) with ~0.5mpp, use 512 x ~0.5
    d = downsample_factor

    if clamp:
        q05, q95 = torch.quantile(scores, clamp), torch.quantile(scores, 1-clamp)
        scores.clamp_(q05,q95)
    
    if norm:
        scores = NormalizeData(scores)

    for (x,y), s in zip(coords, scores):

        # x, y = int(x)*512, int(y)*512
        x, y = x*patch_size, y*patch_size
        
        if colormap == 'RdBu': 
            attention_map[round(y/d):round((y+offset)/d), round(x/d):round((x+offset)/d)] = 1 - s.item()
        else: 
            attention_map[round(y/d):round((y+offset)/d), round(x/d):round((x+offset)/d)] = s.item()
        tissue_map[round(y/d):round((y+offset)/d), round(x/d):round((x+offset)/d)] = s.item()
            
    tissue_map[tissue_map>=0] = 1
    tissue_map[tissue_map<0] = 0

    attention_map_255 = show_cam_on_image(image, attention_map, tissue_map)
    attention_map_255 = np.array(attention_map * 255., dtype=np.uint8)

    # get prediction
    res = -1
    best_th = 0
    ths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    total_iou = 0
    all_ious = []
    for th in ths:
        bi_heatmap = np.ones((attention_map.shape[0],attention_map.shape[1]))
        for i in range(attention_map.shape[0]):
            for j in range(attention_map.shape[1]):
                if slide_path in doub: 
                    if j < 0.5 * attention_map.shape[1]:
                        continue
                if slide_path in doub_left:
                    if j > 0.5 * attention_map.shape[1]:
                        continue
                if (attention_map[i][j]) > th:
                    bi_heatmap[i][j] = 0
        iou = binaryMaskIOU(1-gt, 1-bi_heatmap)
        total_iou += iou
        all_ious.append(iou)
        if iou > res:
            res = iou
            best_th = th
    
    output_pred = np.zeros((image.shape[0],image.shape[1],3), np.uint8)
    output_pred[:] = (255,255,255)
    for i in range(attention_map.shape[0]):
        for j in range(attention_map.shape[1]):
            if slide_path in doub: 
                if j < 0.5 * attention_map.shape[1]:
                    continue
            if slide_path in doub_left:
                if j > 0.5 * attention_map.shape[1]:
                    continue

            if (attention_map[i][j]) > best_th:
                output_pred[i][j] = (0,0,0)

    return attention_map_255, output_gt, output_pred, ths, all_ious

def compute_pixels(graph, scores, slide_gt, slide_root, dataset, patch_size=256, overlay=True, clamp=0.05, norm=True, colormap='jet', crop=False):
    doub = ['C3L-04759-25', 'C3L-02654-21']
    doub_left = ['C3L-02625-22']
    print(dataset)
    if dataset =='cptac':
        if slide_gt == 2:
            tumor_colors = [
                [255, 196, 160],
                [255,0,0],
                [42,0,128],
                [198,233,255],
                [114,188,178],
                [182,0,255],
                [60,128,0],
                [0,243,255],
                [180,162,255]
            ]
        else:
            tumor_colors = [
                [255, 196, 160],
                [0, 255, 0],
                [255,0,0],
            ]      
    else:
        tumor_colors = [
                [0,255,0],
                [255,0,158],
                [130,0,43],
                [60,128,0],
                [0,255,170],
                [0,255,255],
                [105,105,105],
                [255,0,0],
                [150,150,0],
                [255,0,255],
                [36,0,255],
                [255,158,0],
                [182,0,255],
                [255,202,202],
                [198,233,255],
                [197,132,255],
                [255,218,0],
                [105,134,255],
                [0,243,255],
                [114,188,178],
                [255,196,160],
                [0,100,150],
                [255,196,160],
                [255,182,63],
                [180,162,255],
                [152,0,0],
                [14,0,150],
                [255,103,103],
                [0,146,255],
                [255,0,0],
                [130,0,43],
                [42,0,128],
                [255,0,158],
                [255,158,0],
                [36,0,255],
                [0,255,0],
                [60,128,0],
                [182,0,255],
                [198,233,255],
                [0,243,255],
                [114,188,178],
                [180,162,255],
            ]   
    if dataset == 'pcga':
        f = open('../datasets/PCGA/measurements.json')
        pcga_crop = json.load(f)
        if graph.slide_path[0] in pcga_crop:
            Cx, Cy = float(pcga_crop[graph.slide_path[0]]['Cx']), float(pcga_crop[graph.slide_path[0]]['Cy'])
            A = float(pcga_crop[graph.slide_path[0]]['A'])
            P = float(pcga_crop[graph.slide_path[0]]['P'])

            # Solve for width and height
            w = (P/2 + math.sqrt((P/2)**2 - 4*A)) / 2
            h = A / w

            # Calculate top-left corner coordinates
            x1 = Cx - w/2
            y1 = Cy - h/2

            # # Print results
            # print(f"Width: {w:.2f}")
            # print(f"Height: {h:.2f}")
            # print(f"Top-left corner: ({x1:.2f}, {y1:.2f})")
            w, h = int(w), int(h)
            x1, y1 = int(x1), int(y1)
            crop = True

    # Fetch patch coords & slide path for the tissue
    slide_path = graph.slide_path[0]
    coords = graph.node_coords[0]
    coords = [(int(x), int(y)) for x,y in coords]
    
    # fetch tissue image at specific downsample
    downsample_factor = 16.0
    image = fetch_slide_image(slide_path, slide_root, patch_size, downsample_factor=downsample_factor)
    image = np.asarray(image.convert("RGB"))
    cv2.imwrite('haha.png', image)
    total_count = 0
    bg_count = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            total_count += 1
            r, g, b = image[i][j]
            if r>230 and g>230 and b>230:
                bg_count += 1
    tis_count = total_count - bg_count
    print(slide_path)
    print(bg_count)
    print(total_count)
    print(tis_count)

    image = (image - image.min()) / (image.max() - image.min())

    y_min, y_max, x_min, x_max = 0, image.shape[0], 0, image.shape[1]

    annotation = Image.open('/SeaExp/Rushin/datasets/{}/Annotations/{}-annotations.png'.format(dataset.upper(), slide_path)).convert('RGB')
    # if crop:
    #     # w_factor = width//w
    #     # h_factor = height//hs
    #     # print(x1, y1, x1+w, y1+h)
    #     #annotation = annotation.crop((x1*2, y1*2, (x1+w)*2, (y1+h)*2))
    #     #annotation = annotation.crop((y1*2,x1*2, (y1+h)*2,(x1+w)*2))
    annotation = annotation.resize((image.shape[1], image.shape[0]))
    annotation = np.array(annotation)

    # generate annotation
    tumor_arr = [0] * len(tumor_colors)

    output_gt = np.zeros((annotation.shape[0],annotation.shape[1],3), np.uint8)
    output_gt[:] = (255,255,255)
    gt = np.ones((image.shape[0],image.shape[1]))
    for i in range(annotation.shape[0]):
        for j in range(annotation.shape[1]):
            pixel = annotation[i][j]
            r, g, b = annotation[i][j]
            for ind, (r_th, g_th, b_th) in enumerate(tumor_colors):
                if r<r_th+10 and r>r_th-10 and g<g_th+10 and g>g_th-10 and b<b_th+10 and b>b_th-10:
                    tumor_arr[ind] = tumor_arr[ind] + 1
                    output_gt[i][j] = (0,0,0)
                    gt[i][j] = 0
    
    tumor_arr = np.array(tumor_arr)
    print(tumor_arr)
    #print(tumor_arr.sum())

    tis_count = (tis_count) * downsample_factor * downsample_factor
    print(tis_count)
    for i in range(len(tumor_arr)):
        tumor_arr[i] = tumor_arr[i] * downsample_factor * downsample_factor
    
    out = str(int(tis_count)) + '\t'
    for num in (tumor_arr):
        out = out + str(int(num)) + '\t'

    return slide_path + '\t' + out