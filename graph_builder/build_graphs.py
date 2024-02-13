import cl as cl

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms
from torch_geometric.utils import dense_to_sparse

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict

from vision_transformer import load_ViT
from swin_transformer import swin_tiny_patch4_window7_224, ConvStem

class ToPIL(object):
    def __call__(self, sample):
        img = sample
        img = transforms.functional.to_pil_image(img)
        return img 

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        img = img.resize((512, 512))

        if self.transform:
            img = self.transform(img)
       
        sample = {'input': img}
        
        return sample 

class ToTensor(object):
    def __call__(self, img):
        # img = sample['input']
        img = VF.to_tensor(img)
        return img
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def save_coords(txt_file, csv_file_path):
    for path in csv_file_path:
        x, y = path.split('/')[-1].split('.')[0].split('_')
        txt_file.writelines(str(x) + '\t' + str(y) + '\n')
    txt_file.close()

def adj_matrix(csv_file_path, output):
    total = len(csv_file_path)
    adj_s = np.zeros((total, total))

    for i in range(total-1):
        path_i = csv_file_path[i]
        x_i, y_i = path_i.split('/')[-1].split('.')[0].split('_')
        for j in range(i+1, total):
            # sptial 
            path_j = csv_file_path[j]
            x_j, y_j = path_j.split('/')[-1].split('.')[0].split('_')
            if abs(int(x_i)-int(x_j)) <=1 and abs(int(y_i)-int(y_j)) <= 1:
                adj_s[i][j] = 1
                adj_s[j][i] = 1

    print("Adjacency matrix not empty: ", np.any(adj_s))


    adj_s = torch.from_numpy(adj_s)
    # adj_s = adj_s.to(device)

    return adj_s

def bag_dataset(args, csv_file_path, im_transforms=None, augmentations=None):

    if im_transforms:
        transform = im_transforms
    else: 
        transform = Compose([ToTensor()])

    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=transform)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def compute_feats(args, bags_list, image_model, im_transforms=None, augment_model=None, save_path=None, whole_slide_path=None, device=torch.device("cpu")):
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        if args.magnification == '20x':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '20.0/*.jpeg'))         
            file_name = bags_list[i].split('/')[-2].rsplit('_', 1)[0]
            print(file_name)

        if args.magnification == '5x' or args.magnification == '10x':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))

        dataloader, bag_size = bag_dataset(args, csv_file_path, im_transforms)
        print('{} files to be processed: {}'.format(len(csv_file_path), file_name))

        if os.path.isdir(os.path.join(save_path, file_name)) or len(csv_file_path) < 1:
            print('alreday exists')
            continue
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().to(device)

                if 'resnet' in args.backbone:
                    feats, _ = image_model(patches)
                elif 'vit' in args.backbone:
                    feats = image_model(patches).to('cpu')
                elif 'ctrans' in args.backbone:
                    feats = image_model(patches).to('cpu')
                #feats = feats.cpu().numpy()
                feats_list.extend(feats)
        
        os.makedirs(os.path.join(save_path, file_name), exist_ok=True)

        txt_file = open(os.path.join(save_path, file_name, 'c_idx.txt'), "w+")
        save_coords(txt_file, csv_file_path)
        # save node features
        output = torch.stack(feats_list, dim=0).to('cpu')
        torch.save(output, os.path.join(save_path, file_name, 'features.pt'))
        # save adjacent matrix ; Check if adjacency matrix is 0. If 0, display the file name.
        adj_s = adj_matrix(csv_file_path, output).to('cpu')
        torch.save(adj_s, os.path.join(save_path, file_name, 'adj_s.pt'))
        # save adjacency matrix as sparse matrix
        adj_s_ei, edge_attr = dense_to_sparse(adj_s)
        torch.save(adj_s_ei, os.path.join(save_path, file_name, 'adj_s_ei.pt'))
        torch.save(edge_attr, os.path.join(save_path, file_name, 'edge_attr.pt'))

        print('\r Computed: {}/{}'.format(i+1, num_bags))
        

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    parser.add_argument('--num_feats', default=512, type=int, help='Feature size')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--dataset', default=None, type=str, help='path to patches')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone')
    parser.add_argument('--magnification', default='20x', type=str, help='Magnification to compute features')
    parser.add_argument('--weights', default=None, type=str, help='path to the pretrained weights')
    parser.add_argument('--augmentations', default=None, type=str, help='options: None ')
    parser.add_argument('--output', default=None, type=str, help='path to the output graph folder')
    args = parser.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    if 'resnet' in args.backbone:
        if args.backbone == 'resnet18':
            resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
            num_feats = 512
        if args.backbone == 'resnet34':
            resnet = models.resnet34(pretrained=False, norm_layer=nn.InstanceNorm2d)
            num_feats = 512
        if args.backbone == 'resnet50':
            resnet = models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
            num_feats = 2048
        if args.backbone == 'resnet101':
            resnet = models.resnet101(pretrained=False, norm_layer=nn.InstanceNorm2d)
            num_feats = 2048
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()
        image_model = cl.IClassifier(resnet, num_feats, output_class=args.num_classes).to(device)

        # load feature extractor
        if args.weights is None:
            print('No feature extractor')
            return
        state_dict_weights = torch.load(args.weights)
        state_dict_init = image_model.state_dict()
        new_state_dict = OrderedDict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        image_model.load_state_dict(new_state_dict, strict=False)
        im_transforms=None
    
    elif 'vit' in args.backbone:
        if args.backbone == 'vits-dino':
            num_feats = 384
            image_model, im_transforms = load_ViT(args.backbone, args.weights, device)
    
    elif 'ctrans' in args.backbone:
        image_model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False).to(device)
        image_model.head = nn.Identity()

        td = torch.load(args.weights)
        image_model.load_state_dict(td['model'], strict=True)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        im_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean = mean, std = std)
            ]
        )

    augment_model = None

    os.makedirs(args.output, exist_ok=True)
    bags_list = glob.glob(args.dataset)
    # bags_list = glob.glob('/SeaExp_1/Rushin/datasets/CPTAC/patches256/*/')
    compute_feats(args, bags_list, image_model, im_transforms=im_transforms, augment_model=augment_model, save_path=args.output, device=device)
    
if __name__ == '__main__':
    main()
