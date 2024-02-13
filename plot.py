import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.optim as optim
import torch.nn.functional as F

import os
import random
from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from gnn import GNN
from util import read_file, separate_data,get_scheduler, find_dataset_using_name

from plotters import create_plotter

def plot(args, model, device, multiple_loaders):
    model.eval()

    # Attach Forward and Backward hooks
    model.zero_grad(set_to_none=True)

    # Hook for activations from final FC layer
    final_fc = model.graph_pred_head[0]
    final_fc_hook = final_fc.register_forward_hook(model.get_layer_activation('final_fc'))

    # initialize a few variables
    y_true = []
    y_pred = []
    plotting_features = []
    plotting_features_dict = {}
    meta_data = pd.DataFrame()
    for loader in multiple_loaders:

        true_labels = list(loader.dataset.classdict.keys())
        to_be_predicted_classes = list(loader.dataset.to_be_predicted_classes.keys())

        dataset_features = []

        for step, graph in enumerate(tqdm(loader, desc="Iteration")):

            graph = graph.to(device)
            dataset_name = loader.dataset.__class__.__name__

            if graph.x.shape[0] == 1:
                pass
            else:
                slide_path = graph.slide_path[0]

                pred = model(graph)
                y_pred = torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu()

                prob = F.softmax(pred.detach(), dim=1)
                prob = prob.squeeze()

                # pooled features
                graph_features = model.layer_acts['final_fc']

                graph_features = torch.flatten(graph_features, 1)

                if loader.dataset.__class__.__name__ == "PcgaDataset":
                    key = 'pcga' 
                elif loader.dataset.__class__.__name__ == "CptacDataset":
                    key = 'cptac'
                elif loader.dataset.__class__.__name__ == "CisDataset":
                    key = 'cis'

                torch.save(graph_features, os.path.join(args.slide_feats_folder[key], slide_path+".pt"))

                sample_info = pd.DataFrame([[graph.slide_path[0], true_labels[graph.y], to_be_predicted_classes[y_pred], prob[y_pred].item(), dataset_name]], \
                                            columns=["Slide_Name", "Ground_Truth", "Hard_Class", "Prob_Confidence", "Dataset_Name"])
                meta_data = pd.concat([meta_data, sample_info], ignore_index=True, axis=0)

    meta_data.to_csv(os.path.join(args.log_path, "model_metadata.csv"))

        #         plotting_features.append(graph_features.squeeze(0).cpu().numpy())
        #         dataset_features.append(graph_features.squeeze(0).cpu().numpy())

        # plotting_features_dict[loader.dataset.__class__.__name__] = dataset_features    

   
    """  # plotting color maps:
    for p in args.plot_functions:
        args.plotter = p
        plotter = create_plotter(args) # create a plotter given opt.plotter and other options

        plotter.set_input(plotting_features, plotting_features_dict, meta_data)     # initialize and set input for plotting. This is a preprocessing step to prepare the data for plotting input.
        plotter.plot()             # regular step: load and print plotting info i.e data used, plotting parameter
    """
    return 0

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbg-ppa data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')

    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--jk', type=str, default='last',
                        help='Jumping knowledge aggregations : last | sum')
    parser.add_argument('--graph_pooling', type=str, default='mean',
                        help='Graph pooling type : sum | mean | max | attention | set2set')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')

    parser.add_argument('--dataset', nargs='+', type=str, default="pcga",
                        help='dataset name (default: pcga | tcga | cis )')
    parser.add_argument('--phase', type=str, default="plot",
                        help='dataset phase : train | test | cams')
    parser.add_argument('--n_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--data_config', type=str, default="simclr_files",
                        help='dataset config i.e tile size and bkg content (default: simclr_8Conn_files)')
    parser.add_argument('--fdim', type=int, default=512,
                        help='expected feature dim for each node.')

    parser.add_argument('--n_folds', type=int, default=5,
                        help='total number of folds.')
    parser.add_argument('--run_name', type=str, default="easy-wind-35",
                        help='run name to get all model logs')                    

    parser.add_argument('--output', type = str, default = "logs", help='Folder in which to save the tsne plots')
    parser.add_argument('--plot_functions', type=str, nargs='+', default=['tsne', 'umap'],
                        help='plot_functions type (default: tsne | umap)')
    parser.add_argument('--cluster_variants', nargs='+', type=str, default="gt",
                        help='dataset name (default: gt)')

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
    
    for fold_idx in range(args.n_folds):
        test_loaders = []
        for item in args.dataset:

            log_path = os.path.join('logs', "{}_fold_{}".format(args.run_name, fold_idx))

            args.slide_feats_folder[item] = os.path.join(log_path, item.upper(), 'slide_features')
            os.makedirs(args.slide_feats_folder[item], exist_ok=True)

            dataset_class = find_dataset_using_name(item)
            print(dataset_class)

            ### automatic dataloading and splitting
            root = os.path.join('/SeaExp/Rushin/datasets', item.upper(), args.data_config)
            wsi_file = os.path.join('/SeaExp/Rushin/datasets', item.upper(), '%s_%s.txt' % (item.upper(), args.phase))
            wsi_ids = read_file(wsi_file)

            test_wsi_ids = wsi_ids

            test_dataset = dataset_class(root, test_wsi_ids, args.fdim, n_classes=args.n_classes, isTrain=False, transform=T.ToSparseTensor(remove_edge_index=False))
            test_loaders.append(DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers))

        # initialize model with trained weights 
        if args.gnn == 'gin':
            model = GNN(gnn_type = 'gin', num_class = test_dataset.num_classes, num_layer = args.num_layer, input_dim = args.fdim, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, JK = args.jk, graph_pooling = args.graph_pooling).to(device)
        else:
            raise ValueError('Invalid GNN type')

        log_path = os.path.join('logs', "{}_fold_{}".format(args.run_name, fold_idx))
        model_load_path = os.path.join(f'{log_path}', f'final_model_{args.run_name}_fold_{fold_idx}.pth')
        model.load_state_dict(torch.load(model_load_path))
        model = model.to(device)
        print("model weights loaded successfully")

        args.output_folder = os.path.join(args.output, args.run_name+"_plot")
        args.fold_idx = fold_idx
        args.log_path = log_path

        plot(args, model, device, test_loaders)

if __name__ == "__main__":
    main()
