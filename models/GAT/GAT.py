import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.nn import GATConv, LayerNorm, TopKPooling
from torch.nn import Linear

"""
Applying pyG lib
"""

import pandas as pd
import numpy as np

# Model Definition
class Conv1DClassifier(nn.Module):
    def __init__(self, input_shape_tuple): # Renamed for clarity
        super(Conv1DClassifier, self).__init__()
        # input_shape_tuple is expected to be (sequence_length, num_input_channels=1)
        # However, Conv1d expects (batch, channels, length).
        # The input_shape_tuple seems to be (length, channels) from usage.
        # Let's assume input_shape_tuple[0] is sequence_length.
        sequence_length = input_shape_tuple[0]
        # in_channels is fixed to 1 as per original code (x.unsqueeze(1))

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2, padding=0) # Default stride is kernel_size (2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2, padding=0) # Default stride is kernel_size (2)
        self.dropout2 = nn.Dropout(0.25)

        # Calculate the size of the flattened features after conv and pool layers
        # Each MaxPool1d(2) halves the length if padding=0 and length is even.
        # If length is odd, it's floor(length/2).
        # For simplicity, assuming sequence_length is divisible by 4.
        # conv_output_size = sequence_length // 4 # This was the original calculation
        
        # More robust calculation for conv_output_size:
        # After conv1 (padding=2, kernel=5): length remains same (L_out = L_in - k + 2p + 1 = L - 5 + 4 + 1 = L)
        # After pool1 (kernel=2, stride=2, padding=0): L_out = floor((L_in - k)/s + 1) = floor((L-2)/2 + 1) = floor(L/2)
        # After conv2 (padding=2, kernel=5): length remains same
        # After pool2 (kernel=2, stride=2, padding=0): L_out = floor((L_pool1_out - 2)/2 + 1) = floor(L_pool1_out/2)
        
        # Let's simulate it:
        l_after_conv1 = sequence_length 
        l_after_pool1 = (l_after_conv1 - 2) // 2 + 1 if l_after_conv1 >=2 else 0 # Simplified: sequence_length // 2
        l_after_conv2 = l_after_pool1
        l_after_pool2 = (l_after_conv2 - 2) // 2 + 1 if l_after_conv2 >=2 else 0 # Simplified: l_after_pool1 // 2
        
        conv_output_size = l_after_pool2 # Final length after all pooling

        self.fc1 = nn.Linear(128 * conv_output_size, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1) # Output 1 logit for BCEWithLogitsLoss

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length)
        if x.dim() == 2:
            x = x.unsqueeze(1) # Reshape to (batch_size, 1, sequence_length) for Conv1D
        
        # x = x.permute(0, 2, 1) # This was commented out, if used, input_shape logic changes.
                                # Current code assumes (batch, channels, length) input to conv1.
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = torch.flatten(x, start_dim=1) # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)  # Output logits, sigmoid will be applied by BCEWithLogitsLoss or manually for preds
        return x    
    

class GATModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, heads, k, add_self_loops, num_layers=3):
        super(GATModel, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.heads = heads
        self.k = k
        self.add_self_loops = add_self_loops
        self.num_layers = num_layers

        # Create GAT layers dynamically
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        if num_layers == 1:
            self.gat_layers.append(
                GATConv(node_feature_dim, hidden_dim, heads=heads, 
                       concat=False, add_self_loops=add_self_loops))
            self.norm_layers.append(LayerNorm(hidden_dim))
        else: 
            # First layer
            self.gat_layers.append(
                GATConv(node_feature_dim, hidden_dim, heads=heads, 
                       add_self_loops=add_self_loops))  # Fixed: Added missing parenthesis
            self.norm_layers.append(LayerNorm(heads * hidden_dim))
            # Intermediate layers
            for _ in range(1, num_layers - 1):
                self.gat_layers.append(
                    GATConv(heads * hidden_dim, hidden_dim, heads=heads, 
                           add_self_loops=add_self_loops))
                self.norm_layers.append(LayerNorm(heads * hidden_dim))

            # Final GAT layer
            self.gat_layers.append(
                GATConv(heads * hidden_dim, hidden_dim, heads=heads,
                        concat=False, add_self_loops=add_self_loops))
            self.norm_layers.append(LayerNorm(hidden_dim))
                 
        # MLP layers
        self.lin0 = nn.Linear(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

        self.topk_pool = TopKPooling(hidden_dim, ratio=k)
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Process through all GAT layers
        for i, (conv, norm) in enumerate(zip(self.gat_layers, self.norm_layers)):
            x = conv(x, edge_index, edge_attr)
            x = norm(x, batch)
            
            # Don't apply ReLU and dropout after last layer
            if i < len(self.gat_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.drop, training=self.training)

        # 2. Readout layer
        x = self.topk_pool(x, edge_index, edge_attr, batch=batch)[0]
        x = torch.transpose(x, 0, 1)
        x = nn.Linear(x.shape[1],
                      batch[-1] + 1, bias=False, 
                     device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))(x)
        x = torch.transpose(x, 0, 1)

        # 3. Apply MLP classifier
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.lin0(x)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        
        z = x  # extract last layer features
        x = self.lin(x)
        
        return x, z