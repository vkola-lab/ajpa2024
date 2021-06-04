import math
import numpy as np
import torch
import cv2

from skimage.color import rgb2lab
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from ssn_pytorch.lib.ssn.ssn import sparse_ssn_iter


@torch.no_grad()
def inference(image, nspix, n_iter, fdim=None, color_scale=0.26, pos_scale=2.5, weight=None, enforce_connectivity=True):
    """
    generate superpixels

    Args:
        image: numpy.ndarray
            An array of shape (h, w, c)
        nspix: int
            number of superpixels
        n_iter: int
            number of iterations
        fdim (optional): int
            feature dimension for supervised setting
        color_scale: float
            color channel factor
        pos_scale: float
            pixel coordinate factor
        weight: state_dict
            pretrained weight
        enforce_connectivity: bool
            if True, enforce superpixel connectivity in postprocessing

    Return:
        labels: numpy.ndarray
            An array of shape (h, w)
    """
    if weight is not None:
        from ssn_pytorch.model import SSNModel
        model = SSNModel(fdim, nspix, n_iter).to("cuda")
        model.load_state_dict(torch.load(weight))
        model.eval()
    else:
        model = lambda data: sparse_ssn_iter(data, nspix, n_iter)

    height, width = image.shape[:2]

    nspix_per_axis = int(math.sqrt(nspix))
    pos_scale = pos_scale * max(nspix_per_axis/height, nspix_per_axis/width)

    coords = torch.stack(torch.meshgrid(torch.arange(height, device="cuda"), torch.arange(width, device="cuda")), 0)
    coords = coords[None].float()

    image = cv2.bilateralFilter(image, 9, 75, 75)
    image = rgb2lab(image)
    image = torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda").float()

    inputs = torch.cat([color_scale*image, pos_scale*coords], 1)

    Q, H, feat = model(inputs)

    print(H.shape, H.layout)
    print(feat.shape, feat.layout)

    print(feat.permute(0,2,1))

    labels = H.reshape(height, width).to("cpu").detach().numpy()
    labels = np.where(labels == 0, labels.max()+1, labels)

    print(np.unique(labels.reshape(-1)))

    if enforce_connectivity:
        segment_size = height * width / nspix
        min_size = int(0.1 * segment_size)
        max_size = int(5.0 * segment_size)
        labels = _enforce_label_connectivity_cython(
            labels[None], min_size, max_size)[0]

    print(np.unique(labels.reshape(-1)))

    return labels


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    from skimage.measure import regionprops, regionprops_table
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="/path/to/image")
    parser.add_argument("--weight", default=None, type=str, help="/path/to/pretrained_weight")
    parser.add_argument("--fdim", default=20, type=int, help="embedding dimension")
    parser.add_argument("--niter", default=10, type=int, help="number of iterations for differentiable SLIC")
    parser.add_argument("--nspix", default=49, type=int, help="number of superpixels")
    parser.add_argument("--color_scale", default=0.26, type=float)
    parser.add_argument("--pos_scale", default=2.5, type=float)
    args = parser.parse_args()

    image = plt.imread(args.image)

    s = time.time()
    label = inference(image, args.nspix, args.niter, args.fdim, args.color_scale, args.pos_scale, args.weight)
    print(f"time {time.time() - s}sec")

    r = regionprops_table(label, properties=['label', 'centroid'])

    plt.figure()

    # for props in zip(r['centroid-0'], r['centroid-1']):
    #     cy, cx = props
    #     plt.plot(cx, cy, 'ro')

    plt.imshow(mark_boundaries(image, label))

    r['label'][-1] = 0
    # print(r)
    # plt.show()
    df = pd.DataFrame(r).sort_values(by=['label']).reset_index(drop=True)
    coords = np.array(list(zip(df['centroid-0'], df['centroid-1'])))

    print(df.head())
    print(coords)

    from scipy.spatial import Delaunay
    tri = Delaunay(coords)
    plt.show()
