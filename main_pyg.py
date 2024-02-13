import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import pandas as pd
import random
import shutil

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, average_precision_score

import torch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch.optim as optim
import torch.nn.functional as F
import wandb

from gnn import GNN
from util import read_file, separate_data, get_scheduler, find_dataset_using_name, EarlyStopping, BinaryCrossEntropyLoss
from evaluate import Evaluator, plot_confusion_matrix

# multicls_criterion = torch.nn.CrossEntropyLoss()
multicls_criterion = BinaryCrossEntropyLoss()
# os.environ['WANDB_DISABLED'] = 'True'

def train(config, epoch, model, device, loader, optimizer, scheduler, train_evaluator):
    model.train()

    train_losses = []
    y_true = []
    y_pred = []
    y_prob = [] 
    pred = None
    for i, graphs in enumerate(tqdm(loader)):

        step = len(loader) * epoch + i

        graphs = graphs.to(device)

        if graphs.x.shape[0] == 1 or graphs.batch[-1] == 0:
            pass
        else:
            pred = model(graphs)
            optimizer.zero_grad()

            loss = multicls_criterion(pred.to(torch.float32), graphs.y.view(-1,))
            # add flooding here
            loss = (loss-config.b).abs() + config.b
            loss.backward()
            optimizer.step()

            wandb.log({'mini-batch-loss/train': loss})

            train_losses.append(loss.item())

        y_true.append(graphs.y.view(-1,1).detach().cpu())
        y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
        y_prob.append(F.softmax(pred.detach(), dim=1).cpu())

    scheduler.step()

    avg_loss = torch.mean(torch.tensor(train_losses))
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    y_prob = torch.cat(y_prob, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    return train_evaluator.eval(input_dict), avg_loss
    # return avg_loss

def eval(model, device, loader, evaluator):
    model.eval()

    val_losses = []
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

                loss = multicls_criterion(pred.to(torch.float32), graphs.y.view(-1,))
                val_losses.append(loss.item())

            y_true.append(graphs.y.view(-1,1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim = 1).view(-1,1).cpu())
            y_prob.append(F.softmax(pred.detach(), dim=1).cpu())
            # print(y_prob[-1])

    avg_loss = torch.mean(torch.tensor(val_losses))

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    y_prob = torch.cat(y_prob, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}

    return evaluator.eval(input_dict), avg_loss

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
    parser.add_argument('--batch_size', type=int, default=8,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=65,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers (default: 0)')
    parser.add_argument('--b', type=float, default=0.1,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_policy', type=str, default='step',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=30,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--l2_weight_decay', type=float, default=0.01,
                        help='The weight decay for L2 Norm in Adam optimizer')

    parser.add_argument('--dataset', type=str, default="tcga",
                        help='dataset name (default: tcga)')
    parser.add_argument('--phase', type=str, default="train",
                        help='dataset phase : train | test | plot')
    parser.add_argument('--n_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--data_config', type=str, default="ctranspath_files",
                        help='dataset config i.e tile size and bkg content (default: simclr_8Conn_files)')
    parser.add_argument('--fdim', type=int, default=768,
                        help='expected feature dim for each node.')

    parser.add_argument('--n_folds', type=int, default=5,
                        help='total number of folds.')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation.  Should be less then 10.')
    parser.add_argument('--no_val', action='store_true', help='no validation set for tuning')
    
    parser.add_argument('--config_file', type=str, default="configs/config.yaml",
                        help='parameter and hyperparameter config i.e all values for model and dataset parameters')
    parser.add_argument('--project_name', type=str, default=None,
                        help='parameter and hyperparameter config i.e all values for model and dataset parameters')

    args = parser.parse_args()

    if args.project_name is None:
        # Add the project name as "Graph-Perciever-{}-{}" where {} is the current month in words and date.
        args.project_name = "Graph-Perciever_{}".format(time.strftime("%B-%d"))
        

    # wandb configurations & creating reqd. folders
    wandb.init(project=args.project_name, config=args.config_file)
    # wandb.run.name = 'Aug15' + "_fold_" + str(args.fold_idx) 
    # Add the wandb.run.name as "Graph-Perciever-{}-{}" where {} is the current month in words and date.
    wandb.run.name = "Graph-Perciever_{}".format(time.strftime("%B-%d")) + "_fold_" + str(args.fold_idx)
    wandb.config.update({'fold_idx': args.fold_idx,
                         'run_name': wandb.run.name,
                         'log_path': os.path.join('logs', wandb.run.name),
                         'device': args.device}, allow_val_change=True)

    config = wandb.config
    os.makedirs(config.log_path, exist_ok=True)

    print(config)

    ### set up seeds and gpu device
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    
    ### cuda device settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:" + str(config.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    root = os.path.join('/SeaExp/Rushin/datasets', config.dataset.upper(), config.data_config)

    wsi_file = os.path.join('/SeaExp/Rushin/datasets', config.dataset.upper(), '%s_%s.txt' % (config.dataset.upper(), config.phase))
    wsi_ids = read_file(wsi_file)

    train_val_ids, test_ids, train_val_labels = separate_data(wsi_ids, config.seed, config.n_folds, config.fold_idx)

    dataset_class = find_dataset_using_name(config.dataset)
    isTrain = True if config.phase == 'train' else False

    ###################### compute maximum number of nodes in dataset ######################
    """  
    args.max_nodes = 0
    full_dataset = dataset_class(root, wsi_ids, config.fdim, config.n_classes, isTrain=isTrain)
    for i in range(len(full_dataset)):
        data = full_dataset[i]
        args.max_nodes = max(args.max_nodes, data.num_nodes)   

    print("Max nodes in dataset: ", args.max_nodes)  
    """
    ########################################################################################

    if config.no_val:
        train_dataset = dataset_class(root, train_val_ids, config.fdim, config.n_classes, isTrain=isTrain, transform=T.ToSparseTensor(remove_edge_index=False))
        np.savetxt(os.path.join(config.log_path, f'{config.run_name}_fold_{config.fold_idx}_train.txt'), train_val_ids, fmt='%s')
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)

    else:
        print("Use Train, Val, Test CV")
        train_ids, valid_ids = train_test_split(train_val_ids, stratify=train_val_labels, random_state=config.seed, test_size=0.25)
        
        train_dataset = dataset_class(root, train_ids, config.fdim, config.n_classes, isTrain=isTrain, transform=T.ToSparseTensor(remove_edge_index=False))
        np.savetxt(os.path.join(config.log_path, f'{config.run_name}_fold_{config.fold_idx}_train.txt'), train_ids, fmt='%s')
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)

        valid_dataset = dataset_class(root, valid_ids, config.fdim, config.n_classes, isTrain=isTrain, transform=T.ToSparseTensor(remove_edge_index=False))
        np.savetxt(os.path.join(config.log_path, f'{config.run_name}_fold_{config.fold_idx}_val.txt'), valid_ids, fmt='%s')
        valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    test_dataset = dataset_class(root, test_ids, config.fdim, config.n_classes, isTrain=isTrain, transform=T.ToSparseTensor(remove_edge_index=False))
    np.savetxt(os.path.join(config.log_path, f'{config.run_name}_fold_{config.fold_idx}_test.txt'), test_ids, fmt='%s')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers)

    # evaluation objects
    train_evaluator = Evaluator(train_dataset)
    valid_evaluator = Evaluator(valid_dataset) 
    test_evaluator = Evaluator(test_dataset)
 
    # model loading
    model = GNN(gnn_type = config.gnn, num_class = train_dataset.num_classes, num_layer = config.num_layer, input_dim = config.fdim, emb_dim = config.emb_dim, drop_ratio = config.drop_ratio, JK = config.jk, graph_pooling = config.graph_pooling).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, 
                            weight_decay=config.l2_weight_decay, 
                            amsgrad=False)

    scheduler = get_scheduler(optimizer, config)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=5, delta=0.02, verbose=True, path=os.path.join(config.log_path, 'checkpoint.pth'))

    if isTrain:
        val_auc_log, val_recall_log = 0, 0
        best_loss_epoch, best_recall_epoch, best_auc_epoch = 0,0,0
        val_loss_log = float('inf')

        wandb.watch(model)

        for epoch in range(config.n_epochs+1):
            print("=====Epoch {}".format(epoch))
            print("Train Loader length", len(train_loader))
            
            # logs loss per iteration and returns avg loss per epoch
            train_perf, train_loss = train(config, epoch, model, device, train_loader, optimizer, scheduler, train_evaluator)

            if not config.no_val:
                print('Evaluating...')
                valid_perf, valid_loss = eval(model, device, valid_loader, valid_evaluator)

                if epoch > 10:
                    if np.mean(valid_perf['rocauc']) > val_auc_log:
                        val_auc_log = np.mean(valid_perf['rocauc'])
                        best_auc_epoch = epoch
                        best_model_save_path = os.path.join(config.log_path, f'best_rocauc_model_{config.run_name}.pth')
                        torch.save(model.state_dict(), best_model_save_path)
                    if np.mean(valid_perf['recall']) > val_recall_log:
                        val_recall_log = np.mean(valid_perf['recall'])
                        best_recall_epoch = epoch
                        best_model_save_path = os.path.join(config.log_path, f'best_recall_model_{config.run_name}.pth')
                        torch.save(model.state_dict(), best_model_save_path)
                    if valid_loss < val_loss_log:
                        val_loss_log = valid_loss
                        best_loss_epoch = epoch
                        best_model_save_path = os.path.join(config.log_path, f'best_loss_model_{config.run_name}.pth')
                        torch.save(model.state_dict(), best_model_save_path)                  

                # save model named by epoch every 10 epochs
                if epoch % 10 == 0:
                    model_save_path = os.path.join(config.log_path, f'epoch_{epoch}_model_{config.run_name}.pth')
                    torch.save(model.state_dict(), model_save_path)

            # print('Train', train_perf)
            print('Validation', valid_perf)

            metrics = {'loss/train': train_loss,
                    'rocauc/train': train_perf['rocauc'],
                    'recall/train': np.mean(train_perf['recall']),
                    'loss/val': valid_loss,                   
                    'rocauc/valid': valid_perf['rocauc'],
                    'recall/valid': np.mean(valid_perf['recall']),
                    'epoch': epoch,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    }

            wandb.log(metrics)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    wandb.log({f'gradient/{name}': wandb.Histogram(param.grad.data.cpu().numpy())})

            # write a snippet for early stopping when the val_loss doesn't change with a delta of 0.02 for 5 epochs
            if epoch > 30:
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if config.no_val:
            torch.save(model.state_dict(), os.path.join(config.log_path, f'best_model_{config.run_name}.pth'))
        else:
            print('Saving Final Model')
            final_model_save_path = os.path.join(config.log_path, f'final_model_{config.run_name}.pth')
            torch.save(model.state_dict(), final_model_save_path)

            print(f'Fold {config.fold_idx} - best epoch: {best_auc_epoch} with Val AUROC: {val_auc_log}')
            shutil.copy2(os.path.join(config.log_path, f'best_rocauc_model_{config.run_name}.pth'), os.path.join(config.log_path, f'best_rocauc_model_{config.run_name}_epoch{best_auc_epoch}.pth'))

            print(f'Fold {config.fold_idx} - best epoch: {best_recall_epoch} with Val Recall: {val_recall_log}')
            shutil.copy2(os.path.join(config.log_path, f'best_recall_model_{config.run_name}.pth'), os.path.join(config.log_path, f'best_recall_model_{config.run_name}_epoch{best_recall_epoch}.pth'))

            print(f'Fold {config.fold_idx} - best epoch: {best_loss_epoch} with Val Loss: {val_loss_log}')
            shutil.copy2(os.path.join(config.log_path, f'best_loss_model_{config.run_name}.pth'), os.path.join(config.log_path, f'best_loss_model_{config.run_name}_epoch{best_loss_epoch}.pth'))
        print('Final epoch validation score: {}'.format(valid_perf))

    valid_curve = []
    train_curve = []
    test_curve = []
    valid_cm_plots = []
    test_cm_plots = []
    
    print('Finished Training.....start testing')
    testing_metrics = ['rocauc', 'recall', 'loss']

    for metric in testing_metrics:
        # test model on test set (in-domain)
        best_model_load_path = os.path.join(config.log_path, f'best_{metric}_model_{config.run_name}.pth')
        
        model.load_state_dict(torch.load(best_model_load_path))
        model = model.to(device)
        
        test_perf, test_loss = eval(model, device, test_loader, test_evaluator)
        print('Final evaluation with best {} - Test scores: {}'.format(metric, test_perf))

        final_cm_plot = plot_confusion_matrix(test_perf['cm'], list(test_dataset.classdict.keys()), title='fold{}(Test accuracy={:0.2f})'.format(config.fold_idx+1, np.mean(test_perf['acc'])))
        wandb.log({"{}/ConfusionMatrix".format(metric): final_cm_plot})

        with open(os.path.join(config.log_path, config.run_name+'_best_{}_final_test_perf.txt'.format(metric)), 'w') as f:
            for key, value in test_perf.items():
                f.write('%s:%s\n' % (key, value))

    wandb.finish()
if __name__ == "__main__":
    main()
