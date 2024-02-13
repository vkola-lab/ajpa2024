import torch
from torch_geometric.nn import MessagePassing
import torch_geometric
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GCNConv, TransformerConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm, InstanceNorm
from torch_geometric.utils import degree
from torch.nn.init import kaiming_normal_

import math

class MLP(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim, bias=False),
            # torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim, bias=False),
            torch.nn.Dropout(dropout)
        )

        # Initialize linear layers with Kaiming normal initialization and batchnorm with constant initialization
        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                kaiming_normal_(m.weight)
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

            elif isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


    def forward(self, x):
        return self.net(x)


### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, input_dim, emb_dim, drop_ratio = 0.5, JK = "last", gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 0:
            raise ValueError("Number of GNN layers must be greater than or equal 0.")

        ### Linear layer to transform input node features into output node embedding dimensionality
        self.node_encoder = torch.nn.Linear(input_dim, emb_dim, bias=False)

        # Initialize linear layers with Kaiming normal initialization
        if isinstance(self.node_encoder, torch.nn.Linear):
            kaiming_normal_(self.node_encoder.weight)
            if isinstance(self.node_encoder, torch.nn.Linear) and self.node_encoder.bias is not None:
                torch.nn.init.zeros_(self.node_encoder.bias)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):

            if gnn_type == 'gin':
                kwargs = {'aggr': 'add'}
                self.convs.append(GINConv(nn=MLP(emb_dim, 2*emb_dim, dropout=self.drop_ratio), **kwargs)) # experiment: train_eps=True
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, emb_dim, improved=True, cached=False))
            elif gnn_type == 'transformer':
                self.convs.append(TransformerConv(emb_dim, emb_dim, bias=False))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(BatchNorm(emb_dim)) # uses torch_geometric batchnorm

        # Initialize batchnorm layers with constant initialization
        for m in self.batch_norms:
            if isinstance(m, (torch_geometric.nn.norm.BatchNorm)):
                torch.nn.init.constant_(m.module.weight, 1)
                torch.nn.init.constant_(m.module.bias, 0)

    def forward(self, batched_data):
        
        x, adj_t, batch = batched_data.x, batched_data.adj_t, batched_data.batch

        ### computing input node embedding
        h_list = [self.node_encoder(x)]

        for layer in range(self.num_layer):

            h = self.convs[layer](h_list[layer], adj_t)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.gelu(h), self.drop_ratio, training = self.training)

            h_list.append(h)
           
        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(1, self.num_layer+1):
                node_representation += h_list[layer]

        return node_representation


if __name__ == "__main__":
    pass
