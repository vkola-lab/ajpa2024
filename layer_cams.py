import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F

import os
import random
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

from gnn import GNN
from visualize import generate_relevance, plot_heat_maps
from evaluate import grad_cam
from util import read_file, find_dataset_using_name

def generate_cams(args, model, device, multiple_loaders, index=None):

    model.eval()

    y_true = []
    y_pred = []

    os.makedirs(args.output_folder, exist_ok=True)

    for loader in multiple_loaders:

        true_labels = list(loader.dataset.classdict.keys())
        to_be_predicted_classes = list(loader.dataset.to_be_predicted_classes.keys())

        for step, graph in enumerate(tqdm(loader, desc="Iteration")):

            graph = graph.to(device)
            slide_name = graph.slide_path[0]
            print(slide_name)
            # print(graph.node_coords)

            if graph.x.shape[0] == 1:
                pass
            else:

                # GENERATE VISUALIZATION :
                transformer_attribution, output, y_pred = generate_relevance(model, graph, index=index)
                if index is not None:
                    y_pred = index

                print("logits: ", output)

                prob = F.softmax(output, dim=1)
                prob = prob.squeeze()

                print("Slide: {}, True Class: {}, Predicted Class: {}(p={:.3f})".format(slide_name, true_labels[graph.y], to_be_predicted_classes[y_pred], prob[y_pred].item()))
                del output

                slide_root = os.path.join('/SeaExp/Rushin/datasets/', args.dataset_name.upper(), 'WSIs')
                plt = plot_heat_maps(graph, scores=transformer_attribution, prob=prob[index], slide_root=slide_root, clamp=0.05, save_path=args.output_folder, overlay=True)
                # Use numpy to save attention_blend image to a file
                # attention_blend = Image.fromarray(attention_blend)
                # attention_blend.save(os.path.join(args.output_folder, "{}_{}(fold{})_cam.png".format(slide_name, to_be_predicted_classes[y_pred], args.fold_idx)))
                
                plt.savefig(os.path.join(args.output_folder, "{}_{}(fold{})_cam.png".format(slide_name, to_be_predicted_classes[y_pred], args.fold_idx)))
                plt.close()
            

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--jk', type=str, default='sum',
                        help='Jumping knowledge aggregations : last | sum')
    parser.add_argument('--graph_pooling', type=str, default='gmt',
                        help='Graph pooling type : sum | mean | max | attention | set2set')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 0)')

    parser.add_argument('--dataset', nargs='+', type=str, default="cptac",
                        help='dataset name (default: pcga | tcga | cis )')
    parser.add_argument('--phase', type=str, default="cams",
                        help='dataset phase : train | test | cams')
    parser.add_argument('--n_classes', type=int, default=3,
                        help='number of classes')
    parser.add_argument('--data_config', type=str, default="ctranspath_files",
                        help='dataset config i.e tile size and bkg content (default: simclr_files)')
    parser.add_argument('--fdim', type=int, default=768,
                        help='expected feature dim for each node.')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='patch_size')

    parser.add_argument('-L', '--ignore-bounds', dest='limit_bounds',
		default=True, action='store_false', help='display entire scan area')


    parser.add_argument('--fold_idx', type=int, default=0,
                        help='The fold to consider.')
    parser.add_argument('--run_name', type=str, default="easy-wind-35",
                        help='run name to get all model logs')

    parser.add_argument('--output', type = str, default = "logs", help='Folder in which to save the tsne plots')
    args = parser.parse_args()

    ### set up seeds and gpu device
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    args.slide_feats_folder = {}

    fold_idx = args.fold_idx
    test_loaders = []
    
    for item in args.dataset:

        args.dataset_name = item

        dataset_class = find_dataset_using_name(item)
        print(dataset_class)

        ### automatic dataloading and splitting
        root = os.path.join('/SeaExp/Rushin/datasets', item.upper(), args.data_config)
        wsi_file = os.path.join('/SeaExp/Rushin/datasets', item.upper(), '%s_%s.txt' % (item.upper(), args.phase))
        wsi_ids = read_file(wsi_file)

        dataset = dataset_class(root, wsi_ids, args.fdim, n_classes=args.n_classes, isTrain=False, transform=T.ToSparseTensor(remove_edge_index=False))
        test_loaders.append(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers))

    log_path = os.path.join('logs', "{}_fold_{}".format(args.run_name, fold_idx))

    model = GNN(gnn_type = 'gin', num_class = dataset.num_classes, num_layer = args.num_layer, input_dim = args.fdim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, JK = args.jk, graph_pooling = args.graph_pooling).to(device)
    
    model.load_state_dict(torch.load(os.path.join(log_path, "final_model_{}_fold_{}.pth".format(args.run_name, fold_idx))))
    model = model.to(device)
    print("model weights loaded successfully")

    args.output_folder = os.path.join(args.output, args.run_name+"_{}_{}".format(args.patch_size, args.phase), args.dataset_name)

    for i in range(0, args.n_classes):
        args.index = i
        print("Generating CAMs for class {}".format(i)) 
        generate_cams(args, model, device, test_loaders, args.index)

if __name__ == "__main__":
    main()
