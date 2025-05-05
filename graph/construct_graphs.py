import pandas as pd
from graph import nodes, edges, structure
from tqdm import tqdm
import numpy as np
from workflow.parameters_setter import ParameterSetter
import torch
from torch_geometric.data import Data



def construct_graphs(workflow_settings: ParameterSetter, data: pd.DataFrame):
    """
    construct_graphs
    :param workflow_settings:
    :param data: List (id, sequence itself, activity, label)
    :return:
        graphs_representations: list of Data
        labels: list of labels
        partition: identification of the old_data partition each instance belongs to
    """
    # nodes
    nodes_features, esm2_contact_maps = nodes.esm2_derived_features(workflow_settings, data)

    # edges
    adjacency_matrices, weights_matrices, data = edges.get_edges(workflow_settings, data, esm2_contact_maps)

    n_samples = len(adjacency_matrices)
    with tqdm(range(n_samples), total=len(adjacency_matrices), desc="Generating graphs", disable=False) as progress:
        graphs = []
        for i in range(n_samples):
            graphs.append(to_parse_matrix(adjacency_matrix=adjacency_matrices[i],
                                          nodes_features=np.array(nodes_features[i], dtype=np.float32),
                                          weights_matrix=weights_matrices[i],
                                          label=data.iloc[i]['activity'] if 'activity' in data.columns else None))
            progress.update(1)

    return graphs, data

def construct_graphs2(workflow_settings: ParameterSetter, data: pd.DataFrame):
    """
    construct_graphs
    :param workflow_settings:
    :param data: List (id, sequence itself, activity, label)
    :return:
        graphs_representations: list of Data
        labels: list of labels
        partition: identification of the old_data partition each instance belongs to
    """
    # nodes
    #nodes_features, esm2_contact_maps = nodes.esm2_derived_features(workflow_settings, data)
    esm2_contact_maps
    # edges = []
    adjacency_matrices, weights_matrices, data = edges.get_edges(workflow_settings, data, esm2_contact_maps)

    n_samples = len(adjacency_matrices)
    nodes_features = np.ones(n_samples)
    with tqdm(range(n_samples), total=len(adjacency_matrices), desc="Generating graphs", disable=False) as progress:
        graphs = []
        for i in range(n_samples):
            graphs.append(to_parse_matrix(adjacency_matrix=adjacency_matrices[i],
                                          nodes_features=np.array(nodes_features[i], dtype=np.float32),
                                          weights_matrix=weights_matrices[i],
                                          label=data.iloc[i]['label'] if 'label' in data.columns else None))
            progress.update(1)

    return graphs, data

def to_parse_matrix(adjacency_matrix, nodes_features, weights_matrix, label, eps=1e-6):
    """
    :param label: label
    :param adjacency_matrix: Adjacency matrix with shape (n_nodes, n_nodes)
    :param weights_matrix: Edge matrix with shape (n_nodes, n_nodes, n_edge_features)
    :param nodes_features: node embedding with shape (n_nodes, n_node_features)
    :param eps: default eps=1e-6
    :return:
    """

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

def prepare_data(workflow_settings: ParameterSetter, data: pd.DataFrame):

    """
    Prepare dataset using only node embeddings without graph structures.

    :param workflow_settings: Workflow settings.
    :param data: DataFrame with (id, sequence, activity, label).
    :return:
        features: Tensor of shape (n_samples, n_features)
        labels: Tensor of shape (n_samples,)
    """
    # Extract node features (embeddings)
    print(workflow_settings)
    nodes_features, esm2_contact_maps = nodes.esm2_derived_features(workflow_settings, data)
    # Extract edges features (atom_coordinates_matrices)
    atom_coordinates_matrices, weights_matrices, data = structure.get_edges(workflow_settings, data, esm2_contact_maps)

    max_length_f = max(len(x) for x in nodes_features)  # Find the longest feature vector
    max_length_coor = max(x.shape[0] for x in atom_coordinates_matrices)  # Find the longest feature vector
    
    features = pad_or_truncate(nodes_features, max_length_f)
    coordinates = pad_or_truncate(atom_coordinates_matrices, max_length_coor)
    
    atom_coordinates_matrices_arr = np.array(coordinates, dtype=np.float32)
    
    # Extract labels
    labels = np.array(data["activity"].values, dtype=np.float32) if "activity" in data.columns else None
    #labels = torch.tensor(data["activity"].values, dtype=torch.float32) if "activity" in data.columns else None
    return features, atom_coordinates_matrices_arr, labels
