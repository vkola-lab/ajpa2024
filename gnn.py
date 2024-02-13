import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_

from conv import GNN_node
from aggr import GraphMultisetAggregation

from torch_scatter import scatter_mean, scatter

class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, input_dim = 512, emb_dim = 64,
                    gnn_type = 'gin', drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 0:
            raise ValueError("Number of GNN layers must be greater than 0.")

        ### GNN to generate node embeddings
        self.gnn_node = torch.nn.Sequential(GNN_node(num_layer, input_dim, emb_dim, JK = JK, drop_ratio = 0, gnn_type = gnn_type))
                                            # torch.nn.LayerNorm(emb_dim)) # layernorm for transformer aggregation

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "gmt":
            self.pool = GraphMultisetAggregation(dim=emb_dim, k=200, num_pma_blocks=1, num_encoder_blocks=3, heads=8, dropout=drop_ratio)
        else:
            raise ValueError("Invalid graph pooling type.")

        # graph dense layers
        self.graph_pred_head = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, self.emb_dim, bias=False),
                                                   torch.nn.Linear(self.emb_dim, self.num_class, bias=False))

        # Initialize graph prediction head layers with Kaiming normal initialization
        for m in self.graph_pred_head.modules():
            if isinstance(m, torch.nn.Linear):
                kaiming_normal_(m.weight)
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.layer_grads = {}
        self.layer_acts = {}

    def forward(self, batched_data, register_hook=False):
        
        self.final_conv_acts = self.gnn_node(batched_data)

        edge_index, batch, node_coords = batched_data.edge_index, batched_data.batch, batched_data.node_coords
    
        if self.graph_pooling == "gmt":
            h_graph = self.pool(x=self.final_conv_acts, batch=batch, node_coords=node_coords, register_hook=register_hook)
        else:
            h_graph = self.pool(x=self.final_conv_acts, batch=batch)
        return self.graph_pred_head(h_graph)
    
    def get_layer_activation(self, name):
        # the hook signature
        def hook(model, input, output):
            self.layer_acts[name] = output.detach()
            
        return hook

    def get_layer_gradients(self, name):
        # the hook signature
        def hook(model, grad_input, grad_output):
            self.layer_grads[name] = grad_output[0]

        return hook

if __name__ == '__main__':
    GNN(num_class = 10)
