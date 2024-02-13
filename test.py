import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.optim as optim
import torch.nn.functional as F

import wandb

from gnn import GNN
from util import read_file, separate_data, find_dataset_using_name
from evaluate import Evaluator, plot_confusion_matrix

def eval(model, device, loader, evaluator):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for step, graphs in enumerate(tqdm(loader, desc="Iteration")):

        graphs = graphs.to(device)

        if graphs.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(graphs)

            y_true.append(graphs.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1,1).cpu())
            y_prob.append(F.softmax(pred.detach(), dim=1).cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    y_prob = torch.cat(y_prob, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
    plot_dict = {"y_true": y_true, "y_prob": y_prob}

    test_perf = evaluator.eval(input_dict)

    return test_perf, plot_dict


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
    parser.add_argument('--drop_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--jk', type=str, default='sum',
                        help='Jumping knowledge aggregations : last | sum')
    parser.add_argument('--graph_pooling', type=str, default='gmt',
                        help='Graph pooling type : sum | mean | max | attention | set2set')

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for splitting the dataset into 10 (default: 0)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size for training (default: 1)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')

    parser.add_argument('--dataset', type=str, default="cptac",
                        help='dataset name (default: pcga)')
    parser.add_argument('--phase', type=str, default="test",
                        help='dataset phase : train | test | cams')
    parser.add_argument('--n_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--data_config', type=str, default="ctranspath_files",
                        help='dataset config i.e tile size and bkg content (default: 2048pxTiles_75Bkg)')
    parser.add_argument('--fdim', type=int, default=512,
                        help='expected feature dim for each node.')

    parser.add_argument('--n_folds', type=int, default=1,
                        help='total number of folds.')
    parser.add_argument('--config_file', type=str, default="configs/test_config.yaml",
                        help='parameter and hyperparameter config i.e all values for model and dataset parameters')    
    parser.add_argument('--project_name', type=str, default=None,
                        help='parameter and hyperparameter config i.e all values for model and dataset parameters')                

    
    args = parser.parse_args()
    if args.project_name is None:
        # read the yaml file and get the project name
        if args.config_file is not None:
            import yaml
            try:
                from yaml import CLoader as Loader, CDumper as Dumper
            except ImportError:
                from yaml import Loader, Dumper
            with open(args.config_file, 'r') as f:
                config = yaml.load(f, Loader=Loader)
            args.project_name = config['run_name']['value']
            
    wandb.init(project=args.project_name, config = args.config_file)
    run_name = args.project_name
    wandb.run.name = wandb.config.run_name + "_{}_test".format(wandb.config.dataset)
    wandb.config.update({'run_name': wandb.run.name,
                         'log_path': os.path.join('logs', run_name),
                         'device': args.device}, allow_val_change=True)

    config = wandb.config
    os.makedirs("{}_{}_test".format(config.log_path, wandb.config.dataset), exist_ok=True)

    print(config)

    ### set up seeds and gpu device
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:{}".format(config.device)) if torch.cuda.is_available() else torch.device("cpu")

    print('Evaluating...', config.dataset.upper())

    plot_dicts = [] # plot dicts per fold
    score_dicts = [] # scores per fold
    cm_plots = [] # cm per fold
    for fold_idx in range(config.n_folds):

        ### automatic dataloading and splitting
        dataset_class = find_dataset_using_name(config.dataset)

        if config.dataset == 'tcga':
            wsi_file = os.path.join(f'{config.log_path}_fold_{fold_idx}', f'{run_name}_fold_{fold_idx}_fold_{fold_idx}_test.txt')
            test_ids = read_file(wsi_file)
            test_ids = list(filter(lambda x: x != '\n', test_ids))
            # _, test_ids, _ = separate_data(wsi_ids, config.seed, config.n_folds, fold_idx)

        else:
            wsi_file = os.path.join('/SeaExp/Rushin/datasets', config.dataset.upper(), '{}_{}.txt'.format(config.dataset.upper(), config.phase))
            test_ids = read_file(wsi_file)
    
        root = os.path.join('/SeaExp/Rushin/datasets', config.dataset.upper(), config.data_config)
        test_dataset = dataset_class(root, test_ids, config.fdim, config.n_classes, isTrain=False, transform=T.ToSparseTensor(remove_edge_index=False))
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

        # Load Evaluator based on the dataset
        evaluator = Evaluator(test_dataset)

        # initialize model with trained weights
        model = GNN(gnn_type = config.gnn, num_class = test_dataset.num_classes, num_layer = config.num_layer, input_dim = config.fdim, emb_dim = config.emb_dim, drop_ratio = config.drop_ratio, JK = config.jk, graph_pooling = config.graph_pooling).to(device)
            
        model_load_path = os.path.join(f'{config.log_path}_fold_{fold_idx}', f'final_model_{run_name}_fold_{fold_idx}.pth')
        print(model_load_path)
        model.load_state_dict(torch.load(model_load_path))
        model = model.to(device)

        print("model weights loaded successfully")

        # Perform testing. "test_perf" is a dictionary
        test_perf, plot_dict = eval(model, device, test_loader, evaluator)
        print('Testing', test_perf)

        score_dicts.append(test_perf)
        plot_dicts.append(plot_dict)

        cm_plots.append(plot_confusion_matrix(test_perf['cm'], list(test_dataset.classdict.keys()), title='fold{}(Test accuracy={:0.2f})'.format(fold_idx+1, test_perf['acc'])))

        with open(os.path.join("{}_{}_test".format(config.log_path, wandb.config.dataset), "{}_fold_{}_test_perf.txt".format(run_name, fold_idx)), 'w') as f:
            f.write('Fold {}:\n'.format(fold_idx))
            for key, value in test_perf.items():
                f.write('{}:{}\n'.format(key, value))

    wandb.log({"TestCM_fold_{}".format(idx+1): cm_plots[idx] for idx in range(config.n_folds)})

    test_plot_curves = evaluator.plot_curves(plot_dicts)

    average_scores = evaluator.average_scores(score_dicts)

    print(average_scores)

    wandb.log({"PR_Curve": wandb.Image(test_plot_curves['pr'])})
    wandb.log({"ROC_Curve": wandb.Image(test_plot_curves['roc'])})

    with open(os.path.join("{}_{}_{}".format(config.log_path, wandb.config.dataset, 'test'), '{}_avg_test_perf.txt'.format(run_name)), 'w') as f:
        for key, metric_dict in average_scores.items():
            f.write('{}:\n'.format(key))
            for k, v in metric_dict.items():
                f.write('{}:{}\n'.format(k, v))

    wandb.finish()

if __name__ == "__main__":
    main()
