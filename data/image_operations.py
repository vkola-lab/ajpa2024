import os

import numpy as np
import pandas as pd
import openslide

def read_csv(filename):
    records = pd.read_csv(filename, usecols=['Slide_Name'])
    # filenames = list(records['slide_name'])
    return records

class Slide:

    def __init__(self, data_path, svs_path, record):
        self.slide_name, _ = os.path.splitext(record['Slide_Name'])

        self.slide_path = os.path.join(data_path, self.slide_name)
        self.image = openslide.open_slide(os.path.join(svs_path, self.slide_name+".svs"))
        self.size = self.image.dimensions # w, h

        self.tiles_path = os.path.join(self.slide_path, self.slide_name+'_tiles')

        self.patches = os.listdir(self.tiles_path)
        self.patch_count = len(self.patches)

def downsample_image(slide, downsampling_factor, mode="numpy"):
    """Downsample an Openslide at a factor.
    Takes an OpenSlide SVS object and downsamples the original resolution
    (level 0) by the requested downsampling factor, using the most convenient
    image level. Returns an RGB numpy array or PIL image.
    Args:
        slide: An OpenSlide object.
        downsampling_factor: Power of 2 to downsample the slide.
        mode: String, either "numpy" or "PIL" to define the output type.
    Returns:
        img: An RGB numpy array or PIL image, depending on the mode,
            at the requested downsampling_factor.
        best_downsampling_level: The level determined by OpenSlide to perform the downsampling.
    """

    # Get the best level to quickly downsample the image
    # Add a pseudofactor of 0.1 to ensure getting the next best level
    # (i.e. if 16x is chosen, avoid getting 4x instead of 16x)
    best_downsampling_level = slide.get_best_level_for_downsample(downsampling_factor + 0.1)

    # Get the image at the requested scale
    svs_native_levelimg = slide.read_region((0, 0), best_downsampling_level, slide.level_dimensions[best_downsampling_level])
    target_size = tuple([int(x//downsampling_factor) for x in slide.dimensions])
    img = svs_native_levelimg.resize(target_size)

    # By default, return a numpy array as RGB, otherwise, return PIL image
    if mode == "numpy":
        # Remove the alpha channel
        img = np.array(img.convert("RGB"))

    return img, best_downsampling_level
