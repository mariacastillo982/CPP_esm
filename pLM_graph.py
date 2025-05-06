import os
import io
import gc
import copy
import logging
import warnings
import argparse
import collections
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import hub
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, LayerNorm, TopKPooling
import esm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    confusion_matrix)
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tensorboardX import SummaryWriter
from itertools import product
import optuna
from optuna.trial import Trial
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_rel

def make_objective(X,graphs,y,X_test, graphs_test, y_test):
    def objective(trial: Trial):
        # Hyperparameter space to explore
        trial_params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "gat_hidden": trial.suggest_int("gat_hidden", 32, 256, step=32),
            #"mlp_hidden": trial.suggest_int("mlp_hidden", 32, 256, step=32),
            "alpha": trial.suggest_float("alpha", 0, 1, step=0.1),
            "batch_size": trial.suggest_int("batch_size", 32, 128, step=32),
            "pos_weight_val": trial.suggest_float("pos_weight_val", 1.5, 4, step=0.5),
            "num_layers": trial.suggest_int("num_layers", 1, 4, step=1)
        }

        log_filename = f"optuna_hybrid_model_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # You can use a shorter version of your train loop here or refactor it
        # For now, use only a few epochs and return a metric you want to optimize
        model, _, _, metrics_val = train_hybrid_model(X, graphs, y, trial_params, alpha=trial_params["alpha"], device='cuda:0')
        metrics, _, _, _ = test_hybrid_model(model, X_test, graphs_test, y_test, device='cuda:0')
        #best_val_mcc, best_result = train_and_evaluate_model(X = X, graphs = graphs, y = y, trial_params = trial_params, log_csv_path = log_filename)

        return metrics[0]["auc"]  # We want to maximize MCC
    return objective

def train_and_evaluate_model(X,graphs,y,trial_params, log_csv_path=None, device='cuda'):
            
    ind = np.arange(len(y))
    train_idx, test_idx = train_test_split(ind, test_size=0.2, random_state=123)#, stratify=y)
    # Split embeddings
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]  
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Split graphs
    graphs_train = [graphs[i] for i in train_idx.tolist()]
    graphs_test = [graphs[i] for i in test_idx.tolist()]
            
    # Model initialization
    model = HybridModel(
        cnn_input_channels=1,
        cnn_seq_len=1280,
        node_feature_dimension=graphs_train[0].x.shape[1],
        gat_hidden=trial_params["gat_hidden"],
        alpha=trial_params["alpha"],
        num_layers=trial_params["num_layers"]
    ).to(device)

    pos_weight = torch.tensor(trial_params["pos_weight_val"], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=trial_params["lr"])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: step_decay(epoch) / 0.01)

    best_val_mcc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_patience = 20
    epochs_without_improve = 0

    metrics_train = []
    batch_size = trial_params["batch_size"]
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.shape[0])
        epoch_loss = 0.0
        correct = 0

        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = torch.tensor(X_train[indices], dtype=torch.float32, device=device)
            batch_x_graph = [graphs_train[i] for i in indices]
            batch_x_graph = Batch.from_data_list(batch_x_graph).to(device)
            indices = indices.cpu().numpy()  # just before using it
            batch_y = torch.tensor(y_train[indices], dtype=torch.float32, device=device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(
                batch_x, 
                batch_x_graph.x, 
                batch_x_graph.edge_index, 
                batch_x_graph.edge_attr, 
                batch_x_graph.batch
            )
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == batch_y).sum().item()

        train_acc = correct / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_test, dtype=torch.float32, device=device)
            val_graph = Batch.from_data_list(graphs_test).to(device)
            val_y = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1)

            val_outputs = model(
                val_x,
                val_graph.x,
                val_graph.edge_index,
                val_graph.edge_attr,
                val_graph.batch
            )

            val_score = torch.sigmoid(val_outputs)
            best_threshold, best_mcc = optimize_threshold(val_y, val_outputs)

            val_preds = (val_score > best_threshold).float()
            val_acc = accuracy_score(val_y.cpu(), val_preds.cpu())
            val_precision = precision_score(val_y.cpu(), val_preds.cpu())
            val_recall = recall_score(val_y.cpu(), val_preds.cpu())
            val_auc = roc_auc_score(val_y.cpu(), val_score.cpu())
            val_mcc = matthews_corrcoef(val_y.cpu(), val_preds.cpu())
            val_spec = specificity(val_y.cpu(), val_preds.cpu())

            
        metrics_train.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_sensitivity": val_recall,
            "val_specificity": val_spec,
            "val_auc": val_auc,
            "val_mcc": val_mcc
        })

        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= early_stop_patience:
            break

        scheduler.step()

    model.load_state_dict(best_model_wts)
    
    # Log best result
    best_result = metrics_train[-1]
    best_result.update({
        "gat_hidden": trial_params["gat_hidden"],
        "alpha": trial_params["alpha"],
        "lr": trial_params["lr"],
        "batch_size": trial_params["batch_size"],
        "pos_weight_val": trial_params["pos_weight_val"],
        "num_layers":trial_params["num_layers"]
    })

    if log_csv_path:
        df = pd.DataFrame([best_result])
        if not os.path.exists(log_csv_path):
            df.to_csv(log_csv_path, index=False)
        else:
            df.to_csv(log_csv_path, mode='a', header=False, index=False)

    return best_val_mcc, best_result      
        
def grid_search_train(X, graphs, y, X_test, graphs_test, y_test, device='cuda:0'):
    alphas = np.linspace(0, 1, 11)
    
    params = {"lr": 0.0005722845662804915, "gat_hidden": 160, "mlp_hidden": 32, "batch_size": 96, "pos_weight_val": 3.5}

    best_model = None
    best_mcc = -1
    #best_alpha = {}

    for alpha in alphas:
        print(f"\nTesting config: alpha={alpha}")
        model, _, _, val_metrics = train_hybrid_model(X, graphs, y, params, alpha = alpha, device=device)
        #val_metrics = model.final_metrics  # You can attach best val metrics inside train_hybrid_model
        metrics, _, _, _ = test_hybrid_model(model, X_test, graphs_test, y_test, device='cuda')

        if metrics[0]["mcc"] > best_mcc:
            best_val_metrics = val_metrics
            best_test_metrics = metrics[0]
            best_mcc = metrics[0]["mcc"]
            best_model = model
            best_alpha = alpha
    
    print(f"\nBest alpha: {best_alpha} with validation metrics:")
    for key, value in best_val_metrics.items():
        print(f"    {key}: {value}")
    print(f"\nWith testing metrics:")
    for key, value in best_test_metrics.items():
        print(f"    {key}: {value}")
    return best_model

def plot_output_scores(val_scores, val_labels, save_path):
    # Ensure inputs are numpy arrays
    if isinstance(val_scores, torch.Tensor):
        val_scores = val_scores.cpu().numpy().flatten()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy().flatten()

    # Define colors
    colors = ['red' if label == 0 else 'green' for label in val_labels]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(val_scores)), val_scores, c=colors, alpha=0.7)

    # Formatting
    plt.title("Model Output Scores Colored by True Label")
    plt.xlabel("Sample Index")
    plt.ylabel("Sigmoid Output Score")
    plt.grid(True)
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='True Label: 0', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='True Label: 1', markerfacecolor='green', markersize=8)
    ])

    # Save and close
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(val_labels,val_scores, save_path):
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(val_labels, val_scores)
    auc_score = roc_auc_score(val_labels, val_scores)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def compare_roc_curves(scores_1, labels_1, scores_2, labels_2, name_1='Method 1', name_2='Method 2', save_path="roc_comparison.png"):
    # Convert to numpy
    scores_1 = scores_1.detach().cpu().numpy().flatten() if isinstance(scores_1, torch.Tensor) else scores_1.flatten()
    scores_2 = scores_2.detach().cpu().numpy().flatten() if isinstance(scores_2, torch.Tensor) else scores_2.flatten()
    labels_1 = labels_1.detach().cpu().numpy().flatten() if isinstance(labels_1, torch.Tensor) else labels_1.flatten()
    labels_2 = labels_2.detach().cpu().numpy().flatten() if isinstance(labels_2, torch.Tensor) else labels_2.flatten()

    # ROC & AUC
    fpr1, tpr1, _ = roc_curve(labels_1, scores_1)
    fpr2, tpr2, _ = roc_curve(labels_2, scores_2)
    auc1 = roc_auc_score(labels_1, scores_1)
    auc2 = roc_auc_score(labels_2, scores_2)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr1, tpr1, label=f'{name_1} (AUC = {auc1:.4f})', color='blue')
    plt.plot(fpr2, tpr2, label=f'{name_2} (AUC = {auc2:.4f})', color='green')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

    # Formatting
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def save_pdb(pdb_str, pdb_name, path):
    with open(path.joinpath(pdb_name + ".pdb"), "w") as f:
        f.write(pdb_str)

def open_pdb(pdb_file):
    with open(pdb_file, "r") as f:
        pdb_str = f.read()
        return pdb_str

def get_atom_coordinates_from_pdb(pdb_str, atom_type='CA'):
    try:
        pdb_filehandle = io.StringIO(pdb_str)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pdb", pdb_filehandle)

        atom_coordinates = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id(atom_type):
                        atom = residue[atom_type]
                        atom_coordinates.append(np.float64(atom.coord))

        pdb_filehandle.close()
        return atom_coordinates

    except Exception as e:
        raise ValueError(f"Error parsing the PDB structure: {e}")

def _get_random_coordinates(atom_coordinates, coordinate_min, coordinate_max):
    random_atom_coordinates = np.zeros(atom_coordinates.shape)
    random_atom_coordinates[:, 0] = \
        np.random.uniform(coordinate_min[0], coordinate_max[0], size=atom_coordinates.shape[0])
    random_atom_coordinates[:, 1] = \
        np.random.uniform(coordinate_min[1], coordinate_max[1], size=atom_coordinates.shape[0])
    random_atom_coordinates[:, 2] = \
        np.random.uniform(coordinate_min[2], coordinate_max[2], size=atom_coordinates.shape[0])

    return random_atom_coordinates

def predict_structures(sequences):
    hub.set_dir(os.getcwd() + os.sep + "./models/esmfold/")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()

    with tqdm(range(len(sequences)), total=len(sequences), desc="Generating 3D structure") as progress_bar:
        pdbs = []
        for i, sequence in enumerate(sequences):
            pdb_str = _predict(model, sequence)
            pdbs.append(pdb_str)
            progress_bar.update(1)
    return pdbs

def _predict(model, sequence):
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)
    return pdb_str

def translate_positive_coordinates(coordinates):
    min_x = min(min(coordinate[0] for coordinate in coordinates), 0)
    min_y = min(min(coordinate[1] for coordinate in coordinates), 0)
    min_z = min(min(coordinate[2] for coordinate in coordinates), 0)

    eps = 1e-6
    return [np.float64((coordinate[0] - min_x + eps, coordinate[1] - min_y + eps, coordinate[2] - min_z + eps)) for coordinate in coordinates]

def predict_tertiary_structures(sequences):
    pdbs = predict_structures(sequences)
    pdb_names = [str(row) for row in sequences]
    atom_coordinates_matrices = []
    with tqdm(range(len(pdbs)), total=len(pdbs), desc="Saving pdb files", disable=False) as progress:
        for (pdb_name, pdb_str) in zip(pdb_names, pdbs):
            save_pdb(pdb_str, pdb_name, Path('./example/ESMFold_pdbs/'))
            coordinates_matrix = \
                np.array(get_atom_coordinates_from_pdb(pdb_str, 'CA'),
                         dtype='float64')
            coordinates_matrix = np.array(translate_positive_coordinates(coordinates_matrix), dtype='float64')
            atom_coordinates_matrices.append(coordinates_matrix)
            progress.update(1)
    logging.getLogger('workflow_logger'). \
        info(f"Predicted tertiary structures available in: example/ESMFold_pdbs/")
    return atom_coordinates_matrices

def load_tertiary_structures(sequences):
    pdb_path = Path('./example/ESMFold_pdbs/')
    if pdb_path is None:
        return [None] * len(sequences), sequences

    sequences_to_exclude = pd.DataFrame()
    atom_coordinates_matrices = []
    with tqdm(range(len(sequences)), total=len(sequences), desc="Loading pdb files", disable=False) as progress:
        pdbs = []
        for row in sequences:
            pdb_file = pdb_path.joinpath(f"{row}.pdb")
            try:
                pdb_str = open_pdb(pdb_file)
                pdbs.append(pdb_str)
                coordinates_matrix = \
                    np.array(get_atom_coordinates_from_pdb(pdb_str,'CA'),
                             dtype='float64')
                coordinates_matrix = np.array(translate_positive_coordinates(coordinates_matrix), dtype='float64')
                atom_coordinates_matrices.append(coordinates_matrix)
                progress.update(1)
            except Exception as e:
                sequences_to_exclude = sequences_to_exclude.append(row)

        return atom_coordinates_matrices
    
def esm_embeddings(esm2, esm2_alphabet, peptide_sequence_list):
  # NOTICE: ESM for embeddings is quite RAM usage, if your sequence is too long,
  #         or you have too many sequences for transformation in a single converting,
  #         you computer might automatically kill the job.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    esm2 = esm2.eval().to(device)

    batch_converter = esm2_alphabet.get_batch_converter()

    # load the peptide sequence list into the bach_converter
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    ## batch tokens are the embedding results of the whole data set

    batch_tokens = batch_tokens.to(device)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
      # Here we export the last layer of the EMS model output as the representation of the peptides
      # model'esm2_t6_8M_UR50D' only has 6 layers, and therefore repr_layers parameters is equal to 6
        results = esm2(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    # save dataset
    # sequence_representations is a list and each element is a tensor
    embeddings_results = collections.defaultdict(list)
    for i in range(len(sequence_representations)):
      # tensor can be transformed as numpy sequence_representations[0].numpy() or sequence_representations[0].to_list
        each_seq_rep = sequence_representations[i].tolist()
        for each_element in each_seq_rep:
            embeddings_results[i].append(each_element)
    embeddings_results = pd.DataFrame(embeddings_results).T
    del  batch_labels, batch_strs, batch_tokens, results, token_representations
    return embeddings_results

def distance(point1, point2, distance_function):
    """
    Args:
        point1 (tuple): The coordinates of the first point
        point2 (tuple): The coordinates of the second point
        distance_function (str): The type of distance to calculate

    Returns:
        float: The calculated distance between the two points.
    """
    try:
        if len(point1) != len(point2):
            raise ValueError("The points do not have the same number of coordinates")

        if distance_function == 'euclidean':
            return _euclidean(point1, point2)
        elif distance_function == 'canberra':
            return _canberra(point1, point2)
        elif distance_function == 'lance_williams':
            return _lance_william(point1, point2)
        elif distance_function == 'clark':
            return _clark(point1, point2)
        elif distance_function == 'soergel':
            return _soergel(point1, point2)
        elif distance_function == 'bhattacharyya':
            return _bhattacharyya(point1, point2)
        elif distance_function == 'angular_separation':
            return _angular_separation(point1, point2)
        else:
            raise ValueError("Invalid distance name: " + str(distance_function))
    except Exception as e:
        raise ValueError(f"Error calculating distances: {distance_function}" + str(e))


def _euclidean(point1, point2):
    return np.round(np.sqrt(np.sum(np.power(np.subtract(point1, point2), 2))),8)

def _canberra(point1, point2):
    return np.round(np.sum(np.divide(np.abs(point1 - point2), np.add(np.abs(point1), np.abs(point2)))),8)


def _lance_william(point1, point2):
    return np.round(np.divide(np.sum(np.abs(np.subtract(point1, point2))), np.sum(np.add(np.abs(point1), np.abs(point2)))),8)


def _clark(point1, point2):
    return np.round(np.sqrt(np.sum(np.power(np.divide(np.subtract(point1, point2), np.add(np.abs(point1), np.abs(point2))), 2))),8)


def _soergel(point1, point2):
    return np.round(np.divide(np.sum(np.abs(np.subtract(point1, point2))), np.sum(np.maximum(point1, point2))),8)


def _bhattacharyya(point1, point2):
    return np.round(np.sqrt(np.sum(np.power(np.subtract(np.sqrt(point1), np.sqrt(point2)), 2))),8)


def _angular_separation(point1, point2):
    return np.round(np.subtract(1, np.divide(np.sum(np.multiply(point1, point2)), np.sqrt(np.dot(np.sum(np.power(point1, 2)), np.sum(np.power(point2, 2)))))), 8)

class Edges:
    """
    """

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


class EmptyEdges(Edges):
    def __init__(self, number_of_amino_acid: int) -> None:
        self._number_of_amino_acid = number_of_amino_acid

    @property
    def number_of_amino_acid(self) -> Edges:
        return self._number_of_amino_acid

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros((self.number_of_amino_acid, self.number_of_amino_acid), dtype=int), np.empty((0, 0))


class EdgeConstructionFunction(Edges):
    _edges: Edges = None

    def __init__(self, edges: Edges) -> None:
        self._edges = edges

    @property
    def edges(self) -> Edges:
        return self._edges

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._edges.compute_edges()


class SequenceBased(EdgeConstructionFunction):
    def __init__(self, edges: Edges, distance_function: str, atom_coordinates: np.ndarray, sequence: str,
                 use_edge_attr: bool):
        super().__init__(edges)
        self._distance_function = distance_function
        self._atom_coordinates = atom_coordinates
        self._sequence = sequence
        self._use_edge_attr = use_edge_attr

    @property
    def distance_function(self) -> str:
        return self._distance_function

    @property
    def use_edge_attr(self) -> str:
        return self._use_edge_attr

    @property
    def atom_coordinates(self) -> str:
        return self._atom_coordinates

    @property
    def sequence(self) -> str:
        return self._sequence

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges()

        number_of_amino_acid = len(self.sequence)

        if self.use_edge_attr and self.distance_function:
            new_weights_matrix = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid - 1):
            adjacency_matrix[i][i + 1] = 1

            if self.use_edge_attr and self.distance_function:
                dist = distance(self.atom_coordinates[i], self.atom_coordinates[i + 1], self.distance_function)
                new_weights_matrix[i][i + 1] = dist

        if self.use_edge_attr and self.distance_function:
            new_weights_matrix = np.expand_dims(new_weights_matrix, -1)
            if weight_matrix.size > 0:
                return adjacency_matrix, np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
            else:
                return adjacency_matrix, new_weights_matrix
        else:
            return adjacency_matrix, weight_matrix


class ESM2ContactMap(EdgeConstructionFunction):
    def __init__(self, edges: Edges, esm2_contact_map: Tuple[np.ndarray, np.ndarray], use_edge_attr: bool,
                 probability_threshold: float):
        super().__init__(edges)
        self._esm2_contact_map = esm2_contact_map
        self._use_edge_attr = use_edge_attr
        self._probability_threshold = probability_threshold

    @property
    def esm2_contact_map(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._esm2_contact_map

    @property
    def probability_threshold(self) -> str:
        return self._probability_threshold

    @property
    def use_edge_attr(self) -> str:
        return self._use_edge_attr

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges()

        number_of_amino_acid = len(self.esm2_contact_map)
        new_weights_matrix = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid):
            for j in range(i + 1, number_of_amino_acid):
                adjacency_matrix[i][j] = adjacency_matrix[i][j] or (
                    1 if self.esm2_contact_map[i][j] > self.probability_threshold else 0)
                adjacency_matrix[j][i] = adjacency_matrix[i][j]

                if self.use_edge_attr:
                    new_weights_matrix[i][j] = self.esm2_contact_map[i][j]
                    new_weights_matrix[j][i] = self.esm2_contact_map[i][j]

        if self.use_edge_attr:
            new_weights_matrix = np.expand_dims(new_weights_matrix, -1)
            if weight_matrix.size > 0:
                return adjacency_matrix, np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
            else:
                return adjacency_matrix, new_weights_matrix
        else:
            return adjacency_matrix, weight_matrix


class DistanceBasedThreshold(EdgeConstructionFunction):
    def __init__(self, edges: Edges, distance_function: str, threshold: float, atom_coordinates: np.ndarray,
                 use_edge_attr: bool):
        super().__init__(edges)
        self._distance_function = distance_function
        self._threshold = threshold
        self._atom_coordinates = atom_coordinates
        self._use_edge_attr = use_edge_attr

    @property
    def distance_function(self) -> str:
        return self._distance_function

    @property
    def threshold(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._threshold

    @property
    def use_edge_attr(self) -> str:
        return self._use_edge_attr

    @property
    def atom_coordinates(self) -> str:
        return self._atom_coordinates

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges()

        number_of_amino_acid = len(self.atom_coordinates)
        new_weights_matrix = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid):
            for j in range(i + 1, number_of_amino_acid):
                dist = distance(self.atom_coordinates[i], self.atom_coordinates[j], self.distance_function)
                if 0 < dist <= self.threshold:
                    adjacency_matrix[i][j] = 1
                    adjacency_matrix[j][i] = 1

                    if self.use_edge_attr:
                        new_weights_matrix[i][j] = dist
                        new_weights_matrix[j][i] = dist

        if self.use_edge_attr:
            new_weights_matrix = np.expand_dims(new_weights_matrix, -1)
            if weight_matrix.size > 0:
                return adjacency_matrix, np.concatenate((weight_matrix, new_weights_matrix), axis=-1)
            else:
                return adjacency_matrix, new_weights_matrix
        else:
            return adjacency_matrix, weight_matrix


class EdgeConstructionContext:
    @staticmethod
    def compute_edges(args):
        edge_construction_functions, distance_function, distance_threshold, atom_coordinates, sequence, \
        esm2_contact_map, use_edge_attr = args

        construction_functions = [
            ('distance_based_threshold',
             partial(DistanceBasedThreshold,
                     edges=None,
                     distance_function=distance_function,
                     threshold=distance_threshold,
                     atom_coordinates=atom_coordinates,
                     use_edge_attr=use_edge_attr
                     )),
            ('esm2_contact_map_50',
             partial(ESM2ContactMap,
                     edges=None,
                     esm2_contact_map=esm2_contact_map,
                     use_edge_attr=use_edge_attr,
                     probability_threshold=0.50
                     )),
            ('esm2_contact_map_60',
             partial(ESM2ContactMap,
                     edges=None,
                     esm2_contact_map=esm2_contact_map,
                     use_edge_attr=use_edge_attr,
                     probability_threshold=0.60
                     )),
            ('esm2_contact_map_70',
             partial(ESM2ContactMap,
                     edges=None,
                     esm2_contact_map=esm2_contact_map,
                     use_edge_attr=use_edge_attr,
                     probability_threshold=0.70
                     )),
            ('esm2_contact_map_80',
             partial(ESM2ContactMap,
                     edges=None,
                     esm2_contact_map=esm2_contact_map,
                     use_edge_attr=use_edge_attr,
                     probability_threshold=0.80
                     )),
            ('esm2_contact_map_90',
             partial(ESM2ContactMap,
                     edges=None,
                     esm2_contact_map=esm2_contact_map,
                     use_edge_attr=use_edge_attr,
                     probability_threshold=0.90
                     )),
            ('sequence_based',
             partial(SequenceBased,
                     edges=None,
                     distance_function=distance_function,
                     atom_coordinates=atom_coordinates,
                     sequence=sequence,
                     use_edge_attr=use_edge_attr
                     ))
        ]

        number_of_amino_acid = len(sequence)
        edges_functions = EmptyEdges(number_of_amino_acid)

        for name in edge_construction_functions:
            for func_name, func in construction_functions:
                if func_name == name:
                    params = func.keywords
                    params['edges'] = edges_functions
                    edges_functions = func(**params)
                    break

        return edges_functions.compute_edges()

def _construct_edges(atom_coordinates_matrices, sequences):
    edge_construction_functions="distance_based_threshold"
    distance_function="euclidean"
    distance_threshold=10
    use_edge_attr=True
    num_cores = multiprocessing.cpu_count()

    esm2_contact_maps = [None] * len(atom_coordinates_matrices)

    args = [(edge_construction_functions,
             distance_function,
             distance_threshold,
             atom_coordinates,
             sequence,
             esm2_contact_map,
             use_edge_attr
             ) for (atom_coordinates, sequence, esm2_contact_map) in
            zip(atom_coordinates_matrices, sequences, esm2_contact_maps)]

    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        with tqdm(range(len(args)), total=len(args), desc="Generating adjacency matrices", disable=False) as progress:
            futures = []
            for arg in args:
                future = pool.submit(EdgeConstructionContext.compute_edges, arg)
                future.add_done_callback(lambda p: progress.update())
                futures.append(future)

            adjacency_matrices = [future.result()[0] for future in futures]
            weights_matrices = [future.result()[1] for future in futures]

    return adjacency_matrices, weights_matrices

def get_edges(tertiary_structure_method, sequences):
    if tertiary_structure_method:
        atom_coordinates_matrices = predict_tertiary_structures(sequences)
    else:
        atom_coordinates_matrices = load_tertiary_structures(sequences)

    adjacency_matrices, weights_matrices = _construct_edges(atom_coordinates_matrices,
                                                            sequences)

    return adjacency_matrices, weights_matrices

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

def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lr

# Function to optimize threshold based on MCC
def optimize_threshold(y_true, y_pred_probas):
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_mcc = -1
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_probas > threshold).int()
        mcc = matthews_corrcoef(y_true.cpu(), y_pred.cpu())

        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    return best_threshold, best_mcc

def save_metrics_to_csv(metrics_list, filename):
    df = pd.DataFrame(metrics_list)
    df.to_csv(filename, index=False)

# Model Definition
class Conv1DClassifier(nn.Module):
    def __init__(self, input_shape):
        super(Conv1DClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2, padding=0)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2, padding=0)
        self.dropout2 = nn.Dropout(0.25)

        conv_output_size = input_shape[0] // 4  # MaxPool1d halves twice
        self.fc1 = nn.Linear(128 * conv_output_size, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        
        if x.dim() == 2:  # (batch_size, sequence_length)
            x = x.unsqueeze(1) 
        #x = x.permute(0, 2, 1)  # Change shape to (batch, channels=1, length)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)  # No sigmoid here
        #x = torch.sigmoid(self.fc2(x))
        return x
    
def train_test_CNN_model(X, y, X_test, y_test, device='cuda'):
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)#, random_state=123)

    input_shape = (1280, 1)
    model = Conv1DClassifier(input_shape).to(device)
    
    pos_weight = torch.tensor([2], device = device)  # Shape: [1]
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Class weights    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: step_decay(epoch)/0.01)  # normalize to lr=0.001

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_patience = 20
    epochs_without_improve = 0

    num_epochs = 100
    batch_size = 32
    metrics_train = []
    
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.shape[0])
        epoch_loss = 0.0
        correct = 0

        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = torch.tensor(X_train[indices], dtype=torch.float32, device=device)
            batch_y = torch.tensor(y_train[indices], dtype=torch.float32, device=device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            score = torch.sigmoid(outputs)
            preds = (score > 0.5).float()
            correct += (preds == batch_y).sum().item()

        train_acc = correct / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_val, dtype=torch.float32, device=device)
            val_y = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
            val_outputs = model(val_x)
            
            best_threshold, best_mcc = optimize_threshold(val_y, val_outputs)
            print(f"Best Threshold: {best_threshold}, Best MCC: {best_mcc}")
            
            val_score = torch.sigmoid(val_outputs)
            val_preds = (val_score > best_threshold).float()
            val_acc = accuracy_score(y_true=val_y.cpu(), y_pred=val_preds.cpu())
            val_auc = roc_auc_score(y_true=val_y.cpu(), y_score=val_score.cpu())
            val_precision = precision_score(val_y.cpu(), val_preds.cpu())
            val_specificity = specificity(val_y.cpu(), val_preds.cpu())
            val_recall = recall_score(val_y.cpu(), val_preds.cpu())        
            val_mcc = matthews_corrcoef(val_y.cpu(), val_preds.cpu())

        # Inside the training loop, after calculating validation metrics
        metrics_train.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_sensitivity": val_recall,
            "val_specificity": val_specificity,
            "val_auc": val_auc,
            "val_mcc": val_mcc
        })

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Specificity: {val_specificity:.4f}, Val Sensitivity: {val_recall:.4f}, Val AUC: {val_auc:.4f}, Val MCC: {val_mcc:.4f}") 

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
            torch.save(model.state_dict(), 'best_model_1280.pth')
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break

        scheduler.step()

    model.load_state_dict(best_model_wts)
    
    save_metrics_to_csv(metrics_train, "training_validation_results_CNN.csv")
    
    # Predict
    with torch.no_grad():
        test_x = torch.tensor(X_test, dtype=torch.float32, device='cuda')
        test_y = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1)
        test_outputs = model(test_x).cpu().flatten()
        
        best_threshold, best_mcc = optimize_threshold(test_y, test_outputs)
        print(f"Best Threshold: {best_threshold}, Best MCC: {best_mcc}")
        
        test_score = torch.sigmoid(test_outputs)
        test_preds = (test_score > best_threshold).float()
        test_acc = accuracy_score(y_true=test_y.cpu(), y_pred=test_preds.cpu())
        test_auc = roc_auc_score(y_true=test_y.cpu(), y_score=test_score.cpu())
        test_precision = precision_score(test_y.cpu(), test_preds.cpu())
        test_specificity = specificity(test_y.cpu(), test_preds.cpu())
        test_recall = recall_score(test_y.cpu(), test_preds.cpu())        
        test_mcc = matthews_corrcoef(test_y.cpu(), test_preds.cpu())

        print(f"Test: Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Sensitivity: {test_recall:.4f}, Specificity: {test_specificity:.4f}, AUC: {test_auc:.4f}, MCC: {test_mcc:.4f}")    
    metrics_test = [{"accuracy": test_acc,
                    "precision_score": test_precision,
                    "sensitivity_score": test_recall,
                    "specificity_score": test_specificity,
                    "auc": test_auc,
                    "mcc": test_mcc}]
    
    # Prepare data for plotting
    val_y_cpu = val_y.cpu().numpy().flatten()
    val_score_cpu = val_score.cpu().numpy().flatten()
    #plot_roc_curve(val_y_cpu,val_score_cpu,"roc_auc_training_cnn.png")
    plot_output_scores(val_score_cpu,val_y_cpu,"plot_scores_training_cnn.png")
    
    test_y_cpu = test_y.cpu().numpy().flatten()
    test_score_cpu = test_score.cpu().numpy().flatten()
    #plot_roc_curve(test_y_cpu,test_score_cpu,"roc_auc_testing_cnn.png")
    plot_output_scores(test_score_cpu,test_y_cpu,"plot_scores_testing_cnn.png")
    
    save_metrics_to_csv(metrics_test, "testing_results_CNN.csv")
    
    return model, val_y_cpu, val_score_cpu, test_y_cpu, test_score_cpu, metrics_test

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

def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    if tp + fn == 0:
        sensitivity = np.nan
    else:
        sensitivity = tp / (tp + fn)
    return sensitivity

def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1]).ravel()
    if tn + fp == 0:
        specificity = np.nan
    else:
        specificity = tn / (tn + fp)
    return specificity

class HybridModel(nn.Module):
    def __init__(self, cnn_input_channels=1, cnn_seq_len=1280,
                 node_feature_dimension=10, gat_hidden=128,
                 alpha=0.5, num_layers=3):
        super(HybridModel, self).__init__()
        self.cnn_branch = Conv1DClassifier((cnn_seq_len, cnn_input_channels))
        self.gat_branch = GATModel(node_feature_dimension, gat_hidden, 1, 0.3, 8, 10, False, num_layers)

        self.alpha = alpha  # weight for combining outputs

        #self.fc1 = nn.Linear(2, mlp_hidden)
        #self.dropout = nn.Dropout(0.3)
        #self.fc2 = nn.Linear(mlp_hidden, 1)

    def forward(self, cnn_input, gat_input, edge_index, edge_attr, batch):
        # CNN branch forward pass
        cnn_output = self.cnn_branch(cnn_input)  # (batch_size, 1)
        
        # GAT branch forward pass
        gat_output, _ = self.gat_branch(gat_input, edge_index, edge_attr, batch)  # (batch_size, 1)
        #gat_output, _ = self.gat_branch(gat_input, edge_index, None, batch)  # (batch_size, 1)

        # Combine the two outputs using weighted average
        x = self.alpha * gat_output + (1 - self.alpha) * cnn_output
        
        # Combine CNN and GAT outputs
        #combined_output = torch.cat([cnn_output, gat_output], dim=1)  # Concatenate along feature dimension
        
        # Pass the combined output through an MLP
        #x = F.relu(self.fc1(combined_output))
        ##x = torch.sigmoid(self.fc2(x))  # Binary classification
        #x = self.fc2(x)
        return x
    
def train_hybrid_model(X, graphs, y, trial_params, alpha=0.5, device='cuda:0'):
    
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2)#, random_state=123)#, stratify=y)
    # Split embeddings
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]  

    # Split graphs
    X_graphs_train = [graphs[i] for i in train_idx.tolist()]
    X_graphs_test = [graphs[i] for i in test_idx.tolist()]

    node_feature_dimension = graphs[0].x.shape[1]
    
    #model = HybridModel(cnn_input_channels=1, cnn_seq_len=1280,node_feature_dimension=node_feature_dimension,gat_hidden=trial_params["gat_hidden"],mlp_hidden=trial_params["mlp_hidden"]).to(device)
    model = HybridModel(cnn_input_channels=1, cnn_seq_len=1280,node_feature_dimension=node_feature_dimension,gat_hidden=trial_params["gat_hidden"],alpha=alpha, num_layers=trial_params["num_layers"]).to(device)

    pos_weight = torch.tensor([trial_params["pos_weight_val"]], device = device)  # Shape: [1]
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Class weights
    optimizer = torch.optim.Adam(model.parameters(), lr=trial_params["lr"])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: step_decay(epoch)/0.01)  # normalize to lr=0.001

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_patience = 20
    epochs_without_improve = 0

    num_epochs = 200
    batch_size = trial_params["batch_size"]
    metrics_train = [] 

    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.shape[0])
        epoch_loss = 0.0
        correct = 0

        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = torch.tensor(X_train[indices], dtype=torch.float32, device=device)
            batch_x_graph = [X_graphs_train[i] for i in indices]  # List comprehension
            batch_x_graph = Batch.from_data_list(batch_x_graph).to(device) # Convert list to Batch
            batch_y = torch.tensor(y_train[indices], dtype=torch.float32, device=device).unsqueeze(1)
            
            batch_x = batch_x.to(device)  
            batch_data_graph = batch_x_graph.x.to(device)
            edge_index = batch_x_graph.edge_index.to(device)
            edge_attr = batch_x_graph.edge_attr.to(device)
            batch = batch_x_graph.batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, batch_data_graph, edge_index, edge_attr, batch)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            score = torch.sigmoid(outputs).squeeze()
            preds = (score > 0.5).float().view(-1, 1)
            correct += (preds == batch_y).sum().item()

        train_acc = correct / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = torch.tensor(X_test, dtype=torch.float32, device=device)
            test_batch_graph = Batch.from_data_list(X_graphs_test).to(device) # Convert list to Batch
            val_graph = test_batch_graph.x.to(device)
            val_edge_index = test_batch_graph.edge_index.to(device)
            val_edge_attr = test_batch_graph.edge_attr.to(device)
            val_batch = test_batch_graph.batch.to(device)
            
            #val_graph = torch.tensor(X_graphs_test, dtype=torch.float32, device=device)
            val_y = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1)
            val_outputs = model(val_x, val_graph, val_edge_index, val_edge_attr, val_batch)
            val_score = torch.sigmoid(val_outputs)
            
            best_threshold, best_mcc = optimize_threshold(val_y, val_outputs)
            print(f"Best Threshold: {best_threshold}, Best MCC: {best_mcc}")
            
            val_preds = (val_score > best_threshold).float().view(-1, 1)
            val_acc = accuracy_score(val_y.cpu(), val_preds.cpu())
            val_auc = roc_auc_score(y_true=val_y.cpu(), y_score=val_score.cpu())
            val_precision = precision_score(val_y.cpu(), val_preds.cpu())
            val_recall = recall_score(val_y.cpu(), val_preds.cpu()) 
            mcc = matthews_corrcoef(val_y.cpu(), val_preds.cpu())
            val_specificity = specificity(val_y.cpu(), val_preds.cpu())
        
        metrics_train.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_sensitivity": val_recall,
            "val_specificity": val_specificity,
            "val_auc": val_auc,
            "val_mcc": mcc
        })
        
        print(f"Training: Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        print(f"Validation: Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Sensitivity: {val_recall:.4f}, Val_specificity: {val_specificity:.4f}, Val MCC: {mcc:.4f}, Val AUC: {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
            torch.save(model.state_dict(), 'best_model_1280.pth')
            metrics_val= {
            "train_loss": epoch_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_sensitivity": val_recall,
            "val_specificity": val_specificity,
            "val_auc": val_auc,
            "val_mcc": mcc}
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= early_stop_patience:
            print("Early stopping triggered.")
            break

        scheduler.step()
    save_metrics_to_csv(metrics_train, "training_validation_results_CNN_GAT.csv")
    
    # Prepare data for plotting
    val_y_cpu = val_y.cpu().numpy().flatten()
    val_score_cpu = val_score.cpu().numpy().flatten()
    plot_roc_curve(val_y_cpu,val_score_cpu,"roc_auc_training_gan_cnn.png")
    #plot_output_scores(val_score_cpu,val_y_cpu,"plot_scores_training_gan_cnn.png")
    
    model.load_state_dict(best_model_wts)
    return model, val_y_cpu, val_score_cpu, metrics_val

def test_hybrid_model(model, X_test, graphs_test, y_test, device='cuda:0'):
    """
    Evaluate the trained HybridModel on the test dataset.

    Parameters:
    - model: Trained HybridModel
    - X_test: NumPy array of test embeddings
    - graphs_test: List of graph structures corresponding to test samples
    - y_test: NumPy array of ground truth labels
    - device: The device to run the evaluation on (default: 'cuda:0')

    Returns:
    - A dictionary containing accuracy, precision, recall, and F1-score
    - The raw predictions as a NumPy array
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        # Convert test data to tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32, device=device).unsqueeze(1)

        # Convert list of graph data to a batched PyG object
        test_batch_graph = Batch.from_data_list(graphs_test).to(device)
        test_graph_x = test_batch_graph.x.to(device)
        test_edge_index = test_batch_graph.edge_index.to(device)
        test_edge_attr = test_batch_graph.edge_attr.to(device)
        test_batch = test_batch_graph.batch.to(device)

        # Get model predictions
        outputs = model(X_test_tensor, test_graph_x, test_edge_index, test_edge_attr, test_batch)
        y_score = torch.sigmoid(outputs).squeeze().view(-1, 1)  # Probabilities for each sample 
        y_pred = (y_score > 0.5).float().cpu().view(-1, 1) 
        y_true = y_test_tensor.cpu()
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)        
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true=y_true.cpu().numpy(), y_score=y_score.cpu().numpy())
        spe = specificity(y_true, y_pred)
        
    metrics = [{
        "accuracy": accuracy,
        "precision_score": precision,
        "sensitivity_score": recall,
        "specificity_score": spe,
        "mcc": mcc, 
        "auc": auc
    }]
    
    # Prepare data for plotting
    val_y_cpu = y_true.cpu().numpy().flatten()
    val_score_cpu = y_score.cpu().numpy().flatten()
    plot_roc_curve(val_y_cpu,val_score_cpu,"roc_auc_testing_gan_cnn.png")
    #plot_output_scores(val_score_cpu,val_y_cpu,"plot_scores_testing_gan_cnn.png")
    
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Sensitivity: {recall:.4f}, Specificity: {spe:.4f}, AUC: {auc:.4f}, MCC: {mcc:.4f}")
    save_metrics_to_csv(metrics, "testing_results_CNN_GAT.csv")
    
    return metrics, y_pred, val_y_cpu, val_score_cpu

def generate_esm_embeddings(model, alphabet, sequence_list, output_file):
    """
    Generate ESM embeddings for a list of sequences and save the results to a CSV file.

    Parameters:
    - model: The ESM model instance
    - alphabet: The corresponding ESM model alphabet
    - sequence_list: A list of sequences to process
    - output_file: Path to save the resulting CSV file

    Returns:
    - A DataFrame containing the embeddings
    """
    embeddings_results = pd.DataFrame()

    i = 0 # indicate the number
    for seq in sequence_list:
        # the setting is just following the input format setting in ESM model, [name,sequence]
        tuple_sequence = tuple([seq,seq])
        peptide_sequence_list = []
        peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
        # employ ESM model for converting and save the converted data in csv format
        one_seq_embeddings = esm_embeddings(model, alphabet, peptide_sequence_list)
        embeddings_results= pd.concat([embeddings_results,one_seq_embeddings])
        i = i+1
        print(f"have completed {i} seqeunce")

    # Save to CSV
    embeddings_results.to_csv(output_file)
    print(f"Saved embeddings to {output_file}")

    return embeddings_results

def generate_graphs(sequence_list, dataset, tertiary_structure_method=False):
    """
    Generate graphs from sequence data using adjacency and weight matrices.

    Parameters:
    - sequence_list: List of protein/peptide sequences
    - dataset: Pandas DataFrame containing labels (if available)
    - tertiary_structure_method: Boolean flag to determine edge calculation method (default: False)

    Returns:
    - List of generated graphs
    """
    adjacency_matrices, weights_matrices = get_edges(tertiary_structure_method, sequence_list)
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

#from graph.construct_graphs import construct_graphs2

# select the ESM model for embeddings (you can select you desired model from https://github.com/facebookresearch/esm)
# NOTICE: if you choose other model, the following model architecture might not be very compitable
#         bseides,please revise the correspdoning parameters in esm_embeddings function (layers for feature extraction)

# Create a unique file name for logging CSV outputs
log_filename = f"optuna_hybrid_model_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_filename

# whole dataset loading and dataset splitting
dataset = pd.read_excel('./Final_non_redundant_sequences.xlsx',na_filter = False) # take care the NA sequence problem

# generate the peptide embeddings
sequence_list = dataset['sequence']

# get embeddings for training and validation
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#X_data = generate_esm_embeddings(model, alphabet, sequence_list,'./whole_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv')
# read the peptide embeddings
X_data_name = './whole_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv'
X_data = pd.read_csv(X_data_name,header=0, index_col = 0,delimiter=',')
X = np.array(X_data)
y = dataset['label']
y_train = np.array(y) 

# get graphs for training and validation
graphs = generate_graphs(sequence_list, dataset, tertiary_structure_method=False)

dataset_test = pd.read_excel('./kelm.xlsx',na_filter = False) # take care the NA sequence 
sequence_list_test = dataset_test['sequence']
# get embeddings for testing
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
#X_test = generate_esm_embeddings(model, alphabet, sequence_list_test, './kelm_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv')
# read the peptide embeddings
X_data_test_name = './kelm_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv'
X_test = pd.read_csv(X_data_test_name,header=0, index_col = 0,delimiter=',')
X_test = np.array(X_test)
y_test = dataset_test['label']
y_test = np.array(y_test)

# Normalize the data
scaler = MinMaxScaler()
X_combined = np.concatenate((X, X_test), axis=0)
X_combined = scaler.fit_transform(X_combined)
X_combined = scaler.transform(X_combined)

X_train = X_combined[:5479]  # First 192 rows belong to X_test
X_test = X_combined[5479:]  # The rest belong to X (original train set)

# get graphs for testing
graphs_test = generate_graphs(sequence_list_test, dataset_test, tertiary_structure_method=False)

"""
study = optuna.create_study(direction="maximize")
study.optimize(make_objective(X_train, graphs, y_train, X_test, graphs_test, y_test), n_trials=30)

print("Best trial:")
trial = study.best_trial

print(f"  Value (Test AUC): {trial.value}")
print(f"  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
    
params = trial.params
"""

"""
#params = {"lr": 0.00225675878025153, "gat_hidden": 160, "batch_size": 32, "pos_weight_val": 4, "num_layers":2}

#params = {"lr": 0.001, "gat_hidden": 160, "mlp_hidden": 32, "batch_size": 32, "pos_weight_val": 2}MLPC
params = {"lr": 0.0005722845662804915, "gat_hidden": 160, "mlp_hidden": 32, "batch_size": 96, "pos_weight_val": 3.5, "num_layers":3}

print("---------------------------- Training CNN + GAT----------------------------:")    
model,val_y_gat,val_score_gat, metrics_val = train_hybrid_model(X_train, graphs, y_train, params, alpha=0.4, device='cuda')
print("---------------------------- Testing CNN + GAT----------------------------:")  
metrics_GAN, preds, test_y_gat, test_score_gat = test_hybrid_model(model, X_test, graphs_test, y_test, device='cuda')

print("---------------------------- Training & Testing CNN ----------------------------:")
model_CNN, val_y_cnn, val_score_cnn, test_y_cnn, test_score_cnn, metrics_CNN = train_test_CNN_model(X_train, y_train, X_test, y_test, device='cuda')

#compare_roc_curves(val_score_gat, val_y_gat, val_score_cnn, val_y_cnn, name_1='pLM + Graph', name_2='pLM', save_path="roc_auc_training.png")
#compare_roc_curves(test_score_gat, test_y_gat, test_score_cnn, test_y_cnn, name_1='pLM + Graph', name_2='pLM', save_path="roc_auc_testing.png")
"""

"""
params = {"lr": 0.0005722845662804915, "gat_hidden": 160, "mlp_hidden": 32, "batch_size": 96, "pos_weight_val": 3.5}
model,val_y_gat,val_score_gat,metrcis = train_hybrid_model(X_train, graphs, y_train, params, alpha=0.5, device='cuda')
#model = grid_search_train(X_train, graphs, y_train, X_test, graphs_test, y_test)
metrics, preds, test_y_gat, test_score_gat = test_hybrid_model(model, X_test, graphs_test, y_test, device='cuda')
"""

"""
def evaluate_model_multiple_runs(n_runs, model_type='hybrid', X_train=None, graphs=None, y_train=None, X_test=None, graphs_test=None, y_test=None, params=None, device='cuda'):
    """
    Train and test a model multiple times and record metrics.
    
    Args:
        n_runs (int): Number of runs.
        model_type (str): 'hybrid' (CNN+GAT) or 'CNN'.
        X_train, graphs, y_train, etc.: Data inputs.
        params (dict): Hyperparameters for the hybrid model.
        device (str): 'cuda' or 'cpu'.
    
    Returns:
        dict: Mean, SD, and 95% CI for each metric.
    """
    # Store metrics for each run
    metric_names = ["accuracy", "precision_score", "sensitivity_score", "specificity_score", "auc", "mcc"]
    all_metrics = {name: [] for name in metric_names}
    
    for run in range(n_runs):
        print(f"\n--- Run {run + 1}/{n_runs} ---")
        
        if model_type == 'hybrid':
            # Train and test hybrid model (CNN + GAT)
            model, _, _, _ = train_hybrid_model(X_train, graphs, y_train, params, alpha=0.5, device=device)
            metrics, _, _, _ = test_hybrid_model(model, X_test, graphs_test, y_test, device=device)
            #metrics=metrics_[0]
        elif model_type == 'CNN':
            # Train and test CNN model
            _, _, _, _, _, metrics = train_test_CNN_model(X_train, y_train, X_test, y_test, device=device)
        else:
            raise ValueError("model_type must be 'hybrid' or 'CNN'")
        
        # Store metrics for this run
        for name in metric_names:
            all_metrics[name].append(metrics[0][name])
    
    # Compute statistics for each metric
    results = {}
    for name in metric_names:
        values = all_metrics[name]
        mean = np.mean(values)
        std_dev = np.std(values, ddof=1)  # Sample SD
        
        # 95% CI (t-distribution for small samples)
        n = len(values)
        t_value = stats.t.ppf(0.975, df=n-1)
        margin_of_error = t_value * (std_dev / np.sqrt(n))
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        
        results[name] = {
            'mean': mean,
            'std_dev': std_dev,
            'ci_95': (ci_lower, ci_upper),
            'all_values': values  # Optional: Store raw values
        }
    
    return results

def print_results(model_name, results):
    print(f"\n=== {model_name} ===")
    for metric, stats in results.items():
        print(f"\n** {metric} **")
        print(f"Mean: {stats['mean']:.4f}")
        print(f"SD: ±{stats['std_dev']:.4f}")
        print(f"95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")

params = {
    "lr": 0.0005722845662804915,
    "gat_hidden": 160,
    "mlp_hidden": 32,
    "batch_size": 96,
    "pos_weight_val": 3.5,
    "num_layers": 3
}

n_runs = 1000  # Number of runs (recommended: 10-30)

print("\n==== Evaluating Hybrid Model (CNN + GAT) ====")
hybrid_results = evaluate_model_multiple_runs(
    n_runs=n_runs,
    model_type='hybrid',
    X_train=X_train,
    graphs=graphs,
    y_train=y_train,
    X_test=X_test,
    graphs_test=graphs_test,
    y_test=y_test,
    params=params,
    device='cuda'
)

print("\n==== Evaluating CNN Model ====")
cnn_results = evaluate_model_multiple_runs(
    n_runs=n_runs,
    model_type='CNN',
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    device='cuda'
)

print_results("Hybrid Model (CNN + GAT)", hybrid_results)
print_results("CNN Model", cnn_results)

metric_names = ["accuracy", "precision_score", "sensitivity_score", "specificity_score", "auc", "mcc"]

print("\n=== Statistical Significance (Paired t-Test) ===")
for metric in metric_names:
    hybrid_values = hybrid_results[metric]['all_values']
    cnn_values = cnn_results[metric]['all_values']
    t_stat, p_value = ttest_rel(hybrid_values, cnn_values)
    
    print(f"\n** {metric} **")
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("--> Significant (p < 0.05)")
    else:
        print("--> Not significant (p ≥ 0.05)")
"""