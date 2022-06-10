# Histological Profiling

Pre-requisites:

    Linux (...), Mac (...)
    NVIDIA GPU (...)
    Python (3.8), h5py (), matplotlib (), numpy (), opencv-python (), openslide-python (), openslide (), pandas (), pillow (), PyTorch (), scikit-learn (), scipy (), tensorflow (), tensorboardx (), torchvision (), smooth-topk.

WSI Segmentation and Patching

WSI patching and segmentation done using CLAM package. TL;DR: since this project only requires image features for training, it is not necessary to save the actual image patches, the new pipeline rids of this overhead and instead only saves the coordinates of image patches during "patching" and loads these regions on the fly from WSIs during feature extraction. This is significantly faster than other pipelines and usually only takes 1-2s for "patching" and a couple minutes to featurize a WSI.

Check WSI Segmentation and Patching for more info on code run.
