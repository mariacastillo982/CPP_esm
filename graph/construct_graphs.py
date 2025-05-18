import pandas as pd
from graph import edges
from tqdm import tqdm
import numpy as np
#from workflow.parameters_setter import ParameterSetter
import torch
from torch_geometric.data import Data
from pathlib import Path

def generate_graphs(sequence_list, dataset, tertiary_structure_method=False, pdb_path = Path('./output/ESMFold_pdbs/')):
    """
    Generate graphs from sequence data using adjacency and weight matrices.

    Parameters:
    - sequence_list: List of protein/peptide sequences
    - dataset: Pandas DataFrame containing labels (if available)
    - tertiary_structure_method: Boolean flag to determine edge calculation method (default: False)

    Returns:
    - List of generated graphs
    """
    adjacency_matrices, weights_matrices = edges.get_edges(tertiary_structure_method, sequence_list, pdb_path)
    n_samples = len(adjacency_matrices)
    graphs = []

    with tqdm(range(n_samples), total=n_samples, desc="Generating graphs", disable=False) as progress:
        for i in range(n_samples):
            # Create node features (size: sequence length x 10, filled with ones)
            nodes_features = np.ones((len(sequence_list[i]), 10), dtype=np.float32)

            # Extract label if available in dataset
            label = dataset.iloc[i]['label'] if 'label' in dataset.columns else None

            # Convert adjacency matrix and features into graph object
            graph = to_parse_matrix(
                adjacency_matrix=adjacency_matrices[i],
                nodes_features=nodes_features,
                weights_matrix=weights_matrices[i],
                label=label
            )

            graphs.append(graph)
            progress.update(1)

    return graphs

def to_parse_matrix(adjacency_matrix, nodes_features, weights_matrix, label, eps=1e-6):
    num_row, num_col = adjacency_matrix.shape
    rows = []
    cols = []
    e_vec = []

    for i in range(num_row):
        for j in range(num_col):
            if adjacency_matrix[i][j] >= eps:
                rows.append(i)
                cols.append(j)
                if weights_matrix.size > 0:
                    e_vec.append(weights_matrix[i][j])
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(nodes_features, dtype=torch.float32)
    edge_attr = torch.tensor(np.array(e_vec), dtype=torch.float32)
    y = torch.tensor([label], dtype=torch.int64) if label is not None else None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.validate(raise_on_error=True)
    return data


def pad_or_truncate(features, fixed_length):
    """Pads or truncates feature vectors to ensure uniform length."""
    padded_features = []
    
    for feature in features:
        feature = np.array(feature, dtype=np.float32)
        seq_len, feat_size = feature.shape  # Get current shape
        
        if seq_len < fixed_length:  # Pad with zeros if too short
            pad_seq = np.zeros((fixed_length - seq_len, feat_size), dtype=np.float32)
            feature = np.vstack((feature, pad_seq))  # Add padding
        else:  # Truncate if too long
            feature = feature[:fixed_length,:]
        padded_features.append(feature)
    
    p = np.array(padded_features, dtype=np.float32)
    
    return p


