import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEBlock(nn.Module):
    def __init__(self, hidden_dimension, dropout_rate=0.2):
        super().__init__()
        self.graphsage_layer = SAGEConv(hidden_dimension, hidden_dimension)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.norm_layer = nn.LayerNorm(hidden_dimension)

    def forward(self, node_features, edge_index):
        hidden = self.graphsage_layer(node_features, edge_index)
        hidden = F.relu(hidden)
        hidden = self.dropout_layer(hidden)
        hidden = self.norm_layer(hidden + node_features)
        return hidden


class GraphOfThoughtPrunerGraphSAGE(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=256, dropout_rate=0.2, block_count=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dimension, hidden_dimension)
        self.blocks = nn.ModuleList()
        for i in range(block_count):
            self.blocks.append(GraphSAGEBlock(hidden_dimension, dropout_rate))
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dimension, 1)
        )

    def forward(self, node_features, edge_index):
        hidden = F.relu(self.input_layer(node_features))
        for block in self.blocks:
            hidden = block(hidden, edge_index)
        return self.output_layer(hidden).squeeze(-1)