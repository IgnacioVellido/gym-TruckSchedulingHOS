# https://github.com/pyg-team/pytorch_geometric/issues/1008
# http://jduarte.physics.ucsd.edu/capstone-particle-physics-domain/weeks/08-extending.html#graph-datasets
# https://github.com/pyg-team/pytorch_geometric/issues/813
# https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.meta.MetaLayer

import torch
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

class Edge_Model(torch.nn.Module):
    def __init__(self, num_node_attr, num_edge_attr, num_global_attr, hiddens):
        super().__init__()
        # in_channels = (2 * num_node_attr + num_edge_attr)
        in_channels = (2 * num_node_attr + num_edge_attr) + num_global_attr
        out_channels = num_edge_attr

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, out_channels))
        
    def forward(self, src, dst, edge_attr, u, batch):
        """
        src, dest: [E, F_x], where E is the number of edges.
        edge_attr: [E, F_e]
        u: [B, F_u], where B is the number of graphs.
        batch: [E] with max entry B - 1.
        """
        out = torch.cat([src, dst, edge_attr, u[batch]], dim=1)
        # out = torch.cat([src, dst, edge_attr], dim=1)
        return self.edge_mlp(out)   

# ------------------------------------------------------------------------------

class Node_Model(torch.nn.Module):
    def __init__(self, num_node_attr, num_edge_attr, num_global_attr, hiddens):
        super().__init__()
        in_channels = num_node_attr + num_edge_attr # num_node_attr + (out_channels of EdgeModel)
        out_channels = num_node_attr

        self.node_mlp_1 = nn.Sequential(
            nn.Linear(in_channels, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, hiddens))
        
        self.node_mlp_2 = Sequential(
            nn.Linear(hiddens + num_node_attr, hiddens),
            nn.ReLU(),
            nn.Linear(hiddens, out_channels))

    # TODO: append u somehow. If u contains information about current node/time,
    # I want that info to propagate through the nodes
    def forward(self, x, edge_index, edge_attr, u, batch):
        """  
        x: [N, F_x], where N is the number of nodes.
        edge_index: [2, E] with max entry N - 1.
        edge_attr: [E, F_e]
        u: [B, F_u]
        batch: [N] with max entry B - 1.
        """
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)   

# ------------------------------------------------------------------------------

class Global_Model(torch.nn.Module):
    def __init__(self, num_node_attr, num_edge_attr, num_global_attr, hiddens):
        super().__init__()
        # in_channels = num_node_attr + num_edge_attr
        in_channels = num_node_attr + num_global_attr
        out_channels = num_global_attr

        self.global_mlp = nn.Sequential(
            nn.Linear(in_channels, 2*hiddens),
            nn.ReLU(),
            nn.Linear(2*hiddens, 3*hiddens),
            nn.ReLU(),
            nn.Linear(3*hiddens, out_channels))
      
    def forward(self, x, edge_index, edge_attr, u, batch):
        """
        x: [N, F_x], where N is the number of nodes.
        edge_index: [2, E] with max entry N - 1.
        edge_attr: [E, F_e]
        u: [B, F_u]
        batch: [N] with max entry B - 1.
        """
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def build_meta_layer(num_node_attr, num_edge_attr, num_global_attr, hiddens):
     return MetaLayer(
            Edge_Model(num_node_attr, num_edge_attr, num_global_attr, hiddens),
            Node_Model(num_node_attr, num_edge_attr, num_global_attr, hiddens),
            Global_Model(num_node_attr, num_edge_attr, num_global_attr, hiddens)
     )