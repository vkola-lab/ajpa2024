import os
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix
from util import plot_confusion_matrix

from util import read_file
from data.cptac import CPTAC_Nodes, separate_data
from models.graphcnn import GraphCNN

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_dataloader, optimizer, epoch):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    correct_accum = 0
    for idx, batch_graph in enumerate(train_dataloader):

        if idx > total_iters:
            break

        # report
        pbar.set_description('epoch: %d' % (epoch))
        pbar.update(1)

        output = model(batch_graph)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        pred = output.max(1, keepdim=True)[1]

        #compute loss
        loss = criterion(output, labels)

        # compute accuracy
        correct = pred.eq(labels.view_as(pred)).sum().cpu().item()

        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        correct_accum += correct

    pbar.close()

    average_loss = loss_accum/total_iters
    average_acc = correct_accum / float(total_iters * args.batch_size)

    print("loss training: %f accuracy training: %f" % (average_loss, average_acc))


    return average_loss, average_acc

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 3):
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def validate(args, model, device, val_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, val_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in val_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    cm = confusion_matrix(labels.cpu().numpy(), pred.cpu().numpy())

    acc_val = correct / float(len(val_graphs))

    print("accuracy validation: %f" % (acc_val))

    return acc_val, cm

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')

    # dataset arguments
    parser.add_argument('--dataset', type=str, default="CPTAC",
                        help='name of dataset (default: CPTAC)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--n_folds', type=int, default=10,
                        help='total number of folds.')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--workers', type=int, default=4,
                        help='# of worker processes to load and process data from')

    # training based arguments
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--iters_per_epoch', type=int, default=200,
                        help='number of iterations per each epoch (default: 200)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--val_freq', type=int, default=5,
                        help='validate the model after "n" intervals of training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 10 (default: 0)')

    # model based arguments
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--input_dim', type=int, default=20,
                        help='number of input channels (default: 5) options: 5 | 20 | 64 | 512')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"],
                        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true",
                                        help='Whether to learn the epsilon weighting for the center nodes. Does not affect training accuracy though.')

    # visualization and output
    parser.add_argument('--exp_name', type = str, default = "exp01",
                                        help='Experiment name to save the logs')
    parser.add_argument('--outfile', type = str, default = "",
                                        help='output file')
    args = parser.parse_args()

    exp_save_path = os.path.join('log', args.exp_name)
    if not os.path.exists(exp_save_path):
        os.makedirs(exp_save_path)

    writer = SummaryWriter(os.path.join('runs', args.exp_name))

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # read file from the respective data folder.
    root = 'dataset/%s/' % (args.dataset)
    train_file = 'dataset/%s/train_wsi.txt' % (args.dataset)
    wsi_ids = read_file(train_file)

    # test_file = 'dataset/%s/test_wsi.txt' % (args.dataset)
    # test_ids = read_file(test_file)

    if 'CPTAC' in args.dataset:
        fdim = args.input_dim
        train_ids, val_ids = separate_data(wsi_ids, args.seed, args.n_folds, args.fold_idx)
        train_graphs = CPTAC_Nodes(root, train_ids, fdim)
        val_graphs = CPTAC_Nodes(root, val_ids, fdim)

    # print(len(train_graphs), len(val_graphs))

    train_dataloader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False, drop_last=True, collate_fn=lambda data: data)
    val_dataloader = DataLoader(val_graphs, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=False, collate_fn=lambda data: data)

    num_classes = len(train_graphs.classdict)
    # print(num_classes, graphs.classdict)


    model = GraphCNN(args.num_layers, args.num_mlp_layers, args.input_dim, args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    max_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss, avg_acc_train = train(args, model, device, train_dataloader, optimizer, epoch)

        if epoch % args.val_freq == 0:
            acc_val, cm = validate(args, model, device, val_graphs, epoch)
            max_acc = max(max_acc, acc_val)

            cm_plot = plot_confusion_matrix(cm, train_graphs.classdict.keys())
            # '''
            if not args.outfile == "":
                with open(os.path.join(exp_save_path, args.outfile), 'a+') as f:
                    f.write("%f %f %f" % (avg_loss, avg_acc_train, acc_val))
                    f.write("\n")
            print("")
            print(model.eps)

            writer.add_scalar('Accuracy/Val', acc_val, epoch)
            writer.add_figure('Confusion_Matrix', cm_plot, epoch)

        writer.add_scalar('Loss/Train', avg_loss, epoch)
        writer.add_scalar('Accuracy/Train', avg_acc_train, epoch)

    # with open(str(args.dataset)+'acc_results.txt', 'a+') as f:
    #     f.write(str(max_acc) + '\n')

    


if __name__ == '__main__':
    main()
