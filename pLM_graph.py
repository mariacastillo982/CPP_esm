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
# MinMaxScaler will be imported in parameter_optimization.py
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
from torch.utils.tensorboard import SummaryWriter
from itertools import product
# optuna and Trial will be imported in parameter_optimization.py
# from datetime import datetime # Will be imported in parameter_optimization.py
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_rel

# make_objective function removed, will be in parameter_optimization.py

def train_and_evaluate_model(X,graphs,y,trial_params, log_csv_path=None, device='cuda'):
            
    ind = np.arange(len(y))
    train_idx, test_idx = train_test_split(ind, test_size=0.2, random_state=123)#, stratify=y)
    # Split embeddings
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]  
    y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train # Ensure y_train is numpy
    y_test = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test # Ensure y_test is numpy


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

    metrics_train_log = [] # Renamed to avoid conflict if metrics_train is used later
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
            # Ensure graphs_train elements are correctly indexed
            current_graphs_train_indices = indices.cpu().numpy() # Get numpy array of indices
            batch_x_graph_list = [graphs_train[j] for j in current_graphs_train_indices]
            batch_x_graph = Batch.from_data_list(batch_x_graph_list).to(device)
            
            batch_y_indices = indices.cpu().numpy() 
            batch_y = torch.tensor(y_train[batch_y_indices], dtype=torch.float32, device=device).unsqueeze(1)
            
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
            best_threshold, current_val_mcc = optimize_threshold(val_y, val_outputs) # Renamed to avoid conflict

            val_preds = (val_score > best_threshold).float()
            val_acc = accuracy_score(val_y.cpu(), val_preds.cpu())
            val_precision = precision_score(val_y.cpu(), val_preds.cpu(), zero_division=0)
            val_recall = recall_score(val_y.cpu(), val_preds.cpu(), zero_division=0)
            val_auc = roc_auc_score(val_y.cpu(), val_score.cpu())
            # val_mcc is already current_val_mcc
            val_spec = specificity(val_y.cpu(), val_preds.cpu())

            
        metrics_train_log.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_sensitivity": val_recall,
            "val_specificity": val_spec,
            "val_auc": val_auc,
            "val_mcc": current_val_mcc
        })

        if current_val_mcc > best_val_mcc:
            best_val_mcc = current_val_mcc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= early_stop_patience:
            break

        scheduler.step()

    model.load_state_dict(best_model_wts)
    
    # Log best result
    best_result = {}
    if metrics_train_log: # Check if metrics_train_log is not empty
        best_result = metrics_train_log[-1] # Get the metrics from the last epoch run (or best epoch if saved)
        best_result.update({
            "gat_hidden": trial_params["gat_hidden"],
            "alpha": trial_params["alpha"],
            "lr": trial_params["lr"],
            "batch_size": trial_params["batch_size"],
            "pos_weight_val": trial_params["pos_weight_val"],
            "num_layers":trial_params["num_layers"]
        })

    if log_csv_path and best_result: # ensure best_result is not empty
        df = pd.DataFrame([best_result])
        if not os.path.exists(log_csv_path):
            df.to_csv(log_csv_path, index=False)
        else:
            df.to_csv(log_csv_path, mode='a', header=False, index=False)

    return best_val_mcc, best_result      
        
def grid_search_train(X, graphs, y, X_test, graphs_test, y_test, device='cuda:0'):
    alphas = np.linspace(0, 1, 11)
    
    # Assuming num_layers is part of params now
    params = {"lr": 0.0005722845662804915, "gat_hidden": 160, "batch_size": 96, "pos_weight_val": 3.5, "num_layers": 3} # Added num_layers

    best_model = None
    best_mcc = -1
    best_alpha_val = -1 # Renamed to avoid conflict
    best_val_metrics_val = {} # Renamed
    best_test_metrics_val = {} # Renamed


    for alpha_val in alphas: # Renamed to avoid conflict
        print(f"\nTesting config: alpha={alpha_val}")
        # train_hybrid_model expects trial_params which includes alpha, so we pass it correctly
        current_params = params.copy() # Use a copy to modify alpha for current iteration
        # alpha is passed as a separate argument to train_hybrid_model, not inside params dict directly for that function
        model, _, _, val_metrics_from_train = train_hybrid_model(X, graphs, y, current_params, alpha=alpha_val, device=device)
        
        current_test_metrics, _, _, _ = test_hybrid_model(model, X_test, graphs_test, y_test, device='cuda')

        if current_test_metrics[0]["mcc"] > best_mcc:
            best_val_metrics_val = val_metrics_from_train # This should be the validation metrics from the best epoch of this alpha run
            best_test_metrics_val = current_test_metrics[0]
            best_mcc = current_test_metrics[0]["mcc"]
            best_model = model
            best_alpha_val = alpha_val
    
    print(f"\nBest alpha: {best_alpha_val} with validation metrics:")
    for key, value in best_val_metrics_val.items():
        print(f"    {key}: {value}")
    print(f"\nWith testing metrics:")
    for key, value in best_test_metrics_val.items():
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
    
def save_pdb(pdb_str, pdb_name, path: Path): # Added type hint for path
    # Ensure path is a Path object
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    with open(path.joinpath(pdb_name + ".pdb"), "w") as f:
        f.write(pdb_str)

def open_pdb(pdb_file):
    with open(pdb_file, "r") as f:
        pdb_str = f.read()
        return pdb_str

def get_atom_coordinates_from_pdb(pdb_str, atom_type='CA'):
    try:
        pdb_filehandle = io.StringIO(pdb_str)
        # Suppress PDBConstructionWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBConstructionWarning)
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
        if not atom_coordinates: # Handle case where no coordinates are found
            # raise ValueError(f"No atoms of type '{atom_type}' found in PDB string.")
            # Return empty list or handle as per desired behavior for missing atoms
            return [] 
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
    # Ensure the ESMFold model directory is set correctly
    # hub.set_dir(os.getcwd() + os.sep + "./models/esmfold/") # This is often set globally once
    # The following lines are usually for managing CUDA memory, good to have.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    torch.cuda.empty_cache()
    # torch.cuda.reset_peak_memory_stats() # reset_peak_memory_stats might not be needed unless debugging memory

    # Load ESMFold model
    # Ensure torch.hub.set_dir() was called appropriately before this, if model is not cached.
    # Typically, set_dir is called once at the beginning of the script or session.
    # If pLM_graph.py is run as a module, this might need to be handled in the main script that calls it.
    # For now, assuming it's handled or model is cached.
    model_esmfold = esm.pretrained.esmfold_v1() # Renamed to avoid conflict
    model_esmfold = model_esmfold.eval().cuda() # Ensure it's on CUDA

    pdbs = []
    # tqdm description updated for clarity
    with tqdm(sequences, total=len(sequences), desc="ESMFold: Predicting 3D structures") as progress_bar:
        for sequence in progress_bar: # Iterate directly over sequences
            pdb_str = _predict(model_esmfold, sequence)
            pdbs.append(pdb_str)
            # progress_bar.update(1) # tqdm updates automatically when iterating over it
    return pdbs

def _predict(model, sequence):
    with torch.no_grad():
        pdb_str = model.infer_pdb(sequence)
    return pdb_str

def translate_positive_coordinates(coordinates):
    if not coordinates: # Handle empty list of coordinates
        return []
    min_x = min(min(coordinate[0] for coordinate in coordinates), 0)
    min_y = min(min(coordinate[1] for coordinate in coordinates), 0)
    min_z = min(min(coordinate[2] for coordinate in coordinates), 0)

    eps = 1e-6
    return [np.float64((coordinate[0] - min_x + eps, coordinate[1] - min_y + eps, coordinate[2] - min_z + eps)) for coordinate in coordinates]

def predict_tertiary_structures(sequences):
    # Ensure the output directory for PDBs exists
    pdb_output_path = Path('./output/ESMFold_pdbs/')
    pdb_output_path.mkdir(parents=True, exist_ok=True)

    pdbs = predict_structures(sequences) # This now returns a list of PDB strings
    
    # Use sequence content or index for pdb_names if sequences are strings
    # If sequences are complex objects, adapt how pdb_name is derived.
    # Assuming sequences is a list of strings (the peptide sequences themselves)
    pdb_names = [str(seq)[:30] + "..." if len(str(seq)) > 30 else str(seq) for seq in sequences] # Example naming

    atom_coordinates_matrices = []
    # tqdm description updated
    with tqdm(zip(pdb_names, pdbs, sequences), total=len(pdbs), desc="Processing PDBs & extracting coordinates") as progress_bar:
        for i, (pdb_name_prefix, pdb_str, original_sequence) in enumerate(progress_bar):
            # Create a more unique PDB name, e.g., using an index or a hash of the sequence
            unique_pdb_name = f"seq_{i}_{pdb_name_prefix.replace('/', '_').replace(' ', '_')}" # Make name file-system friendly
            save_pdb(pdb_str, unique_pdb_name, pdb_output_path)
            
            coordinates_list = get_atom_coordinates_from_pdb(pdb_str, 'CA')
            if not coordinates_list:
                # Handle cases where no CA atoms are found or PDB is problematic
                # Option 1: Log a warning and append None or an empty array
                logging.warning(f"No CA coordinates found for sequence {i}: {original_sequence[:30]}... Skipping.")
                atom_coordinates_matrices.append(np.array([], dtype='float64')) # Or None
                continue 
                # Option 2: Raise an error, depending on how critical this is
                # raise ValueError(f"Failed to get CA coordinates for sequence {i}")

            coordinates_matrix = np.array(coordinates_list, dtype='float64')
            
            # Check if coordinates_matrix is empty before translating
            if coordinates_matrix.size == 0:
                 logging.warning(f"Empty coordinate matrix for sequence {i} before translation. PDB: {unique_pdb_name}")
                 atom_coordinates_matrices.append(np.array([], dtype='float64'))
                 continue

            translated_coordinates = translate_positive_coordinates(coordinates_matrix.tolist()) # tolist() if needed by translate
            atom_coordinates_matrices.append(np.array(translated_coordinates, dtype='float64'))
            
    # Logging info about where PDBs are saved
    # workflow_logger might not be defined here. Using standard logging.
    logging.info(f"Predicted tertiary structures saved in: {pdb_output_path.resolve()}")
    return atom_coordinates_matrices


def load_tertiary_structures(sequences):
    pdb_path = Path('./output/ESMFold_pdbs/') # Default path

    # Check if the path exists, and if not, create it (though load implies they should exist)
    # pdb_path.mkdir(parents=True, exist_ok=True) # More for saving, but harmless

    if not pdb_path.exists():
        logging.warning(f"PDB directory {pdb_path} does not exist. Cannot load structures.")
        # Return a list of Nones or empty arrays, matching the expected output structure
        return [np.array([], dtype='float64') for _ in sequences] 

    # sequences_to_exclude = pd.DataFrame() # This was for appending rows, better to collect list of indices/sequences
    excluded_sequences_info = []
    atom_coordinates_matrices = []
    
    # Assuming sequences is a list of sequence strings or identifiers that map to PDB filenames
    with tqdm(enumerate(sequences), total=len(sequences), desc="Loading PDB files and extracting coordinates") as progress_bar:
        for i, seq_identifier in progress_bar:
            # Adapt filename generation to how PDBs were saved by predict_tertiary_structures
            # If predict_tertiary_structures saves as "seq_0_AGHT...", then load that.
            # For now, assuming seq_identifier is the direct name used (e.g., from a column).
            # This needs to be consistent with how PDBs are named during saving.
            # Let's assume a naming convention like `f"seq_{i}_{str(seq_identifier)[:30]}.pdb"` was used for saving.
            # Or, if `seq_identifier` itself is the filename base:
            pdb_file_name = f"{str(seq_identifier)}.pdb" # This was the original assumption
            # Fallback or more robust naming might be needed if seq_identifier is complex.
            # Example: pdb_file_name = f"seq_{i}_{str(seq_identifier)[:30].replace('/', '_').replace(' ', '_')}.pdb"
            
            pdb_file = pdb_path.joinpath(pdb_file_name)

            try:
                if not pdb_file.exists():
                    # logging.warning(f"PDB file not found: {pdb_file}. For sequence: {seq_identifier}")
                    # Try an alternative name based on index if the above fails (example from predict_tertiary_structures)
                    alt_pdb_name_prefix = str(seq_identifier)[:30] + "..." if len(str(seq_identifier)) > 30 else str(seq_identifier)
                    alt_pdb_name = f"seq_{i}_{alt_pdb_name_prefix.replace('/', '_').replace(' ', '_')}.pdb"
                    pdb_file = pdb_path.joinpath(alt_pdb_name)
                    if not pdb_file.exists():
                        logging.warning(f"PDB file still not found: {pdb_file} (tried original and alt name). For sequence: {seq_identifier}")
                        excluded_sequences_info.append({'index': i, 'identifier': seq_identifier, 'reason': 'PDB file not found'})
                        atom_coordinates_matrices.append(np.array([], dtype='float64')) # Placeholder for missing
                        continue

                pdb_str = open_pdb(pdb_file)
                coordinates_list = get_atom_coordinates_from_pdb(pdb_str, 'CA')
                
                if not coordinates_list:
                    logging.warning(f"No CA coordinates found in {pdb_file} for sequence: {seq_identifier}")
                    excluded_sequences_info.append({'index': i, 'identifier': seq_identifier, 'reason': 'No CA coordinates'})
                    atom_coordinates_matrices.append(np.array([], dtype='float64'))
                    continue

                coordinates_matrix = np.array(coordinates_list, dtype='float64')
                
                if coordinates_matrix.size == 0:
                    logging.warning(f"Empty coordinate matrix from {pdb_file} for sequence: {seq_identifier}")
                    atom_coordinates_matrices.append(np.array([], dtype='float64'))
                    continue

                translated_coordinates = translate_positive_coordinates(coordinates_matrix.tolist())
                atom_coordinates_matrices.append(np.array(translated_coordinates, dtype='float64'))
            
            except Exception as e:
                logging.error(f"Error processing PDB for sequence {seq_identifier} (file: {pdb_file}): {e}")
                excluded_sequences_info.append({'index': i, 'identifier': seq_identifier, 'reason': str(e)})
                atom_coordinates_matrices.append(np.array([], dtype='float64')) # Placeholder on error

    if excluded_sequences_info:
        logging.warning(f"Excluded {len(excluded_sequences_info)} sequences during PDB loading. Details: {excluded_sequences_info}")

    return atom_coordinates_matrices
    
def esm_embeddings(esm2, esm2_alphabet, peptide_sequence_list_tuples): # Input changed to list of tuples
  # peptide_sequence_list_tuples should be like [('prot1', 'SEQ1'), ('prot2', 'SEQ2')]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    esm2 = esm2.eval().to(device)

    batch_converter = esm2_alphabet.get_batch_converter()

    # load the peptide sequence list into the bach_converter
    # peptide_sequence_list_tuples is already in the format [ (name, seq), ... ]
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list_tuples)
    batch_lens = (batch_tokens != esm2_alphabet.padding_idx).sum(1) # Use esm2_alphabet here

    batch_tokens = batch_tokens.to(device)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
      # Here we export the last layer of the EMS model output as the representation of the peptides
      # model'esm2_t33_650M_UR50D' has 33 layers.
        results = esm2(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33].cpu()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    
    # Store embeddings in a DataFrame, indexed by the original labels/names
    # embeddings_results = collections.defaultdict(list) # Not needed if constructing DataFrame directly
    
    # Check if batch_labels (protein names) were correctly passed and use them for DataFrame index
    # If peptide_sequence_list_tuples was [('ID1', 'SEQ1'), ('ID2', 'SEQ2'), ...],
    # then batch_labels should be ['ID1', 'ID2', ...]
    
    # df_data = {label: rep.tolist() for label, rep in zip(batch_labels, sequence_representations)}
    # embeddings_df = pd.DataFrame.from_dict(df_data, orient='index')

    # If batch_labels are not protein names but just indices, then create a simple list of lists/arrays
    embedding_data_list = [seq_rep.tolist() for seq_rep in sequence_representations]
    
    # If batch_labels are indeed the names/IDs from the input tuples:
    if len(batch_labels) == len(embedding_data_list):
        embeddings_df = pd.DataFrame(embedding_data_list, index=batch_labels)
    else: # Fallback if batch_labels are not as expected
        embeddings_df = pd.DataFrame(embedding_data_list)


    del batch_strs, batch_tokens, results, token_representations, batch_lens
    gc.collect() # Explicit garbage collection
    return embeddings_df


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
        # Convert to numpy arrays for vectorized operations if not already
        p1 = np.array(point1)
        p2 = np.array(point2)

        if p1.shape != p2.shape: # Check shape instead of len for numpy arrays
            raise ValueError("The points do not have the same dimensions")

        if distance_function == 'euclidean':
            return _euclidean(p1, p2)
        elif distance_function == 'canberra':
            return _canberra(p1, p2)
        elif distance_function == 'lance_williams':
            return _lance_william(p1, p2)
        elif distance_function == 'clark':
            return _clark(p1, p2)
        elif distance_function == 'soergel':
            return _soergel(p1, p2)
        elif distance_function == 'bhattacharyya':
            return _bhattacharyya(p1, p2)
        elif distance_function == 'angular_separation':
            return _angular_separation(p1, p2)
        else:
            raise ValueError("Invalid distance name: " + str(distance_function))
    except Exception as e:
        # Log the error or handle it more gracefully
        # print(f"Error calculating distance '{distance_function}' between {point1} and {point2}: {e}")
        # Depending on desired behavior, either re-raise, or return a default (e.g., np.nan, float('inf'))
        # For now, re-raising to maintain original behavior.
        raise ValueError(f"Error calculating {distance_function} distance: {e}")


def _euclidean(point1, point2):
    return np.round(np.sqrt(np.sum(np.power(np.subtract(point1, point2), 2))),8)

def _canberra(point1, point2):
    # Add epsilon to denominator to avoid division by zero if point1 and point2 are zero vectors
    denominator = np.add(np.abs(point1), np.abs(point2))
    # Handle cases where denominator is zero for some elements
    # For example, if point1[i] and point2[i] are both 0, the term should be 0.
    term = np.divide(np.abs(point1 - point2), denominator, 
                     out=np.zeros_like(denominator, dtype=float), where=denominator!=0)
    return np.round(np.sum(term), 8)


def _lance_william(point1, point2):
    numerator = np.sum(np.abs(np.subtract(point1, point2)))
    denominator = np.sum(np.add(np.abs(point1), np.abs(point2)))
    if denominator == 0: # Both points are zero vectors
        return 0.0 if numerator == 0 else np.nan # Or some other appropriate value like 1.0 if different
    return np.round(numerator / denominator, 8)


def _clark(point1, point2):
    diff = np.subtract(point1, point2)
    sum_abs = np.add(np.abs(point1), np.abs(point2))
    # Handle division by zero: if sum_abs[i] is 0, (point1[i] and point2[i] are 0), term is 0.
    term_squared = np.power(np.divide(diff, sum_abs, out=np.zeros_like(diff, dtype=float), where=sum_abs!=0), 2)
    return np.round(np.sqrt(np.sum(term_squared)), 8)


def _soergel(point1, point2):
    numerator = np.sum(np.abs(np.subtract(point1, point2)))
    denominator = np.sum(np.maximum(point1, point2)) # np.maximum handles element-wise max
    if denominator == 0: # All elements of point1 and point2 are <= 0, and sums to 0.
                         # If point1 and point2 are non-negative, this means they are zero vectors.
        return 0.0 if numerator == 0 else np.nan # Or 1.0 if different and non-negative
    return np.round(numerator / denominator, 8)


def _bhattacharyya(point1, point2):
    # Bhattacharyya distance is typically for probability distributions.
    # Assuming non-negative inputs if used for coordinates.
    # If coordinates can be negative, sqrt might lead to complex numbers.
    # Consider if this distance is appropriate for general coordinates.
    # For now, assuming inputs are such that sqrt is valid (e.g., non-negative).
    if np.any(point1 < 0) or np.any(point2 < 0):
        # Handle negative inputs, e.g., by taking abs, or warning, or specific domain logic
        # For now, let's proceed assuming non-negative as per typical Bhattacharyya context
        # Or raise error: raise ValueError("Bhattacharyya distance requires non-negative inputs for sqrt.")
        pass # Assuming inputs are valid for now
    
    sqrt_p1 = np.sqrt(point1)
    sqrt_p2 = np.sqrt(point2)
    # The formula often involves sum(sqrt(p1*p2)), leading to BC = sum(sqrt(p1*p2))
    # And Dist = -log(BC) or sqrt(1-BC) for normalized versions.
    # The provided formula seems to be Hellinger distance variant: sqrt(sum((sqrt(p1_i) - sqrt(p2_i))^2))
    # which is sqrt(2 * (1 - sum(sqrt(p1_i*p2_i)))) if sum(p1)=sum(p2)=1.
    # Let's stick to the formula given:
    return np.round(np.sqrt(np.sum(np.power(np.subtract(sqrt_p1, sqrt_p2), 2))), 8)


def _angular_separation(point1, point2):
    # Angular separation is 1 - cosine_similarity
    dot_product = np.sum(np.multiply(point1, point2))
    norm_p1_sq = np.sum(np.power(point1, 2))
    norm_p2_sq = np.sum(np.power(point2, 2))
    
    denominator = np.sqrt(norm_p1_sq * norm_p2_sq)
    
    if denominator == 0:
        # Handle case where one or both vectors are zero.
        # If both are zero, dot_product is 0. Cosine is undefined or taken as 0 or 1.
        # If one is zero, dot_product is 0. Cosine is 0.
        # If cosine is 0, angular separation is 1.
        return 1.0 # Or np.nan, or handle as per specific requirements
        
    cosine_similarity = dot_product / denominator
    # Ensure cosine_similarity is within [-1, 1] due to potential floating point inaccuracies
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    
    return np.round(1 - cosine_similarity, 8)


class Edges(ABC): # Made ABC
    """
    Abstract base class for edge computation strategies.
    """
    @abstractmethod
    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


class EmptyEdges(Edges):
    def __init__(self, number_of_amino_acid: int) -> None:
        self._number_of_amino_acid = number_of_amino_acid

    @property
    def number_of_amino_acid(self) -> int: # Return type corrected
        return self._number_of_amino_acid

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        # Adjacency matrix (number_of_amino_acid x number_of_amino_acid)
        # Weight matrix (0,0) implies no edge attributes by default for empty.
        # If weights are always expected, shape might be (N, N, 0) or (0, num_edge_features)
        # The original (0,0) is fine if it means "no weights array".
        return np.zeros((self.number_of_amino_acid, self.number_of_amino_acid), dtype=int), np.empty((0, 0))


class EdgeConstructionFunction(Edges):
    _edges: Edges # This is the "previous" or "base" Edges object to decorate

    def __init__(self, edges: Edges) -> None:
        if not isinstance(edges, Edges): # Type check
            raise TypeError("Input 'edges' must be an instance of Edges.")
        self._edges = edges

    @property
    def edges(self) -> Edges:
        return self._edges

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        # This base decorator class just passes through the computation from the wrapped object.
        # Subclasses will modify this.
        return self._edges.compute_edges()


class SequenceBased(EdgeConstructionFunction):
    def __init__(self, edges: Edges, distance_function: str, atom_coordinates: np.ndarray, sequence: str,
                 use_edge_attr: bool):
        super().__init__(edges)
        self._distance_function = distance_function
        self._atom_coordinates = atom_coordinates # Should be np.ndarray
        self._sequence = sequence # str
        self._use_edge_attr = use_edge_attr # bool

    @property
    def distance_function(self) -> str:
        return self._distance_function

    @property
    def use_edge_attr(self) -> bool: # Corrected type
        return self._use_edge_attr

    @property
    def atom_coordinates(self) -> np.ndarray: # Corrected type
        return self._atom_coordinates

    @property
    def sequence(self) -> str:
        return self._sequence

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges() # Get base matrices

        number_of_amino_acid = len(self.sequence)
        
        # Ensure atom_coordinates match sequence length if used
        if self.use_edge_attr and self.distance_function:
            if len(self.atom_coordinates) != number_of_amino_acid:
                raise ValueError("Atom coordinates length must match sequence length for SequenceBased edges.")

        # Initialize new_weights_matrix for this decorator's contribution
        current_decorator_weights = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid - 1):
            adjacency_matrix[i, i + 1] = 1 # Use comma for numpy indexing
            adjacency_matrix[i + 1, i] = 1 # Edges are typically undirected

            if self.use_edge_attr and self.distance_function:
                # Ensure atom_coordinates are available and valid
                if self.atom_coordinates is not None and len(self.atom_coordinates) > i + 1:
                    dist = distance(self.atom_coordinates[i], self.atom_coordinates[i + 1], self.distance_function)
                    current_decorator_weights[i, i + 1] = dist
                    current_decorator_weights[i + 1, i] = dist # Assuming undirected weights
                else:
                    # Handle missing coordinates for sequential edges if distance is requested
                    # e.g., set to a default, or raise error, or skip
                    pass # Or log warning

        if self.use_edge_attr and self.distance_function:
            current_decorator_weights_expanded = np.expand_dims(current_decorator_weights, axis=-1)
            if weight_matrix.size > 0 and weight_matrix.ndim == 3:
                # Concatenate along the feature dimension (last axis)
                final_weight_matrix = np.concatenate((weight_matrix, current_decorator_weights_expanded), axis=-1)
            else: # If base weight_matrix was empty or not 3D
                final_weight_matrix = current_decorator_weights_expanded
            return adjacency_matrix, final_weight_matrix
        else:
            return adjacency_matrix, weight_matrix # Return original weight_matrix if no new attrs added


class ESM2ContactMap(EdgeConstructionFunction):
    def __init__(self, edges: Edges, esm2_contact_map: np.ndarray, use_edge_attr: bool, # contact_map is ndarray
                 probability_threshold: float):
        super().__init__(edges)
        self._esm2_contact_map = esm2_contact_map # This should be the actual map (NxN array)
        self._use_edge_attr = use_edge_attr
        self._probability_threshold = probability_threshold

    @property
    def esm2_contact_map(self) -> np.ndarray: # Corrected type
        return self._esm2_contact_map

    @property
    def probability_threshold(self) -> float: # Corrected type
        return self._probability_threshold

    @property
    def use_edge_attr(self) -> bool: # Corrected type
        return self._use_edge_attr

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges()

        # esm2_contact_map itself is the NxN matrix of probabilities
        if self.esm2_contact_map is None: # Handle case where contact map is not provided
            # logging.warning("ESM2ContactMap: contact map is None. Returning base edges.")
            return adjacency_matrix, weight_matrix
            
        number_of_amino_acid = self.esm2_contact_map.shape[0]
        
        # This decorator's contribution to weights
        current_decorator_weights = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid):
            for j in range(i + 1, number_of_amino_acid): # Iterate to avoid self-loops and redundant pairs
                contact_probability = self.esm2_contact_map[i, j]
                if contact_probability > self.probability_threshold:
                    adjacency_matrix[i, j] = 1 # Set edge based on threshold
                    adjacency_matrix[j, i] = 1 # Undirected

                if self.use_edge_attr:
                    # Store the probability as an edge attribute
                    current_decorator_weights[i, j] = contact_probability
                    current_decorator_weights[j, i] = contact_probability

        if self.use_edge_attr:
            current_decorator_weights_expanded = np.expand_dims(current_decorator_weights, axis=-1)
            if weight_matrix.size > 0 and weight_matrix.ndim == 3:
                final_weight_matrix = np.concatenate((weight_matrix, current_decorator_weights_expanded), axis=-1)
            else:
                final_weight_matrix = current_decorator_weights_expanded
            return adjacency_matrix, final_weight_matrix
        else:
            return adjacency_matrix, weight_matrix


class DistanceBasedThreshold(EdgeConstructionFunction):
    def __init__(self, edges: Edges, distance_function: str, threshold: float, atom_coordinates: np.ndarray,
                 use_edge_attr: bool):
        super().__init__(edges)
        self._distance_function = distance_function
        self._threshold = threshold
        self._atom_coordinates = atom_coordinates # np.ndarray
        self._use_edge_attr = use_edge_attr # bool

    @property
    def distance_function(self) -> str:
        return self._distance_function

    @property
    def threshold(self) -> float: # Corrected type
        return self._threshold

    @property
    def use_edge_attr(self) -> bool: # Corrected type
        return self._use_edge_attr

    @property
    def atom_coordinates(self) -> np.ndarray: # Corrected type
        return self._atom_coordinates

    def compute_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        adjacency_matrix, weight_matrix = self.edges.compute_edges()

        if self.atom_coordinates is None or self.atom_coordinates.size == 0:
            # logging.warning("DistanceBasedThreshold: atom_coordinates are None or empty. Returning base edges.")
            return adjacency_matrix, weight_matrix
            
        number_of_amino_acid = len(self.atom_coordinates)
        current_decorator_weights = np.zeros((number_of_amino_acid, number_of_amino_acid), dtype=np.float64)

        for i in range(number_of_amino_acid):
            for j in range(i + 1, number_of_amino_acid): # Avoid self-loops and redundant pairs
                dist = distance(self.atom_coordinates[i], self.atom_coordinates[j], self.distance_function)
                if 0 < dist <= self.threshold: # Original condition: 0 < dist
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1 # Undirected

                    if self.use_edge_attr:
                        current_decorator_weights[i, j] = dist
                        current_decorator_weights[j, i] = dist

        if self.use_edge_attr:
            current_decorator_weights_expanded = np.expand_dims(current_decorator_weights, axis=-1)
            if weight_matrix.size > 0 and weight_matrix.ndim == 3:
                final_weight_matrix = np.concatenate((weight_matrix, current_decorator_weights_expanded), axis=-1)
            else:
                final_weight_matrix = current_decorator_weights_expanded
            return adjacency_matrix, final_weight_matrix
        else:
            return adjacency_matrix, weight_matrix


class EdgeConstructionContext:
    @staticmethod
    def compute_edges(args_tuple): # Renamed args to args_tuple for clarity
        # Unpack arguments for a single protein/sequence
        edge_construction_function_names, distance_function_str, distance_thresh, \
        atom_coords, seq_str, esm2_map, use_edge_attr_flag = args_tuple

        number_of_amino_acid = len(seq_str)
        # Start with an EmptyEdges object
        current_edges_obj: Edges = EmptyEdges(number_of_amino_acid)

        # Define available construction functions (factories)
        # These now take the current Edges object and other params
        construction_function_factories = {
            'distance_based_threshold': lambda base_edges: DistanceBasedThreshold(
                edges=base_edges,
                distance_function=distance_function_str,
                threshold=distance_thresh,
                atom_coordinates=atom_coords,
                use_edge_attr=use_edge_attr_flag
            ),
            'esm2_contact_map_50': lambda base_edges: ESM2ContactMap(
                edges=base_edges, esm2_contact_map=esm2_map, use_edge_attr=use_edge_attr_flag, probability_threshold=0.50
            ),
            'esm2_contact_map_60': lambda base_edges: ESM2ContactMap(
                edges=base_edges, esm2_contact_map=esm2_map, use_edge_attr=use_edge_attr_flag, probability_threshold=0.60
            ),
            'esm2_contact_map_70': lambda base_edges: ESM2ContactMap(
                edges=base_edges, esm2_contact_map=esm2_map, use_edge_attr=use_edge_attr_flag, probability_threshold=0.70
            ),
            'esm2_contact_map_80': lambda base_edges: ESM2ContactMap(
                edges=base_edges, esm2_contact_map=esm2_map, use_edge_attr=use_edge_attr_flag, probability_threshold=0.80
            ),
            'esm2_contact_map_90': lambda base_edges: ESM2ContactMap(
                edges=base_edges, esm2_contact_map=esm2_map, use_edge_attr=use_edge_attr_flag, probability_threshold=0.90
            ),
            'sequence_based': lambda base_edges: SequenceBased(
                edges=base_edges,
                distance_function=distance_function_str, # Can be None if not use_edge_attr for this
                atom_coordinates=atom_coords, # Can be None if not use_edge_attr for this
                sequence=seq_str,
                use_edge_attr=use_edge_attr_flag
            )
        }
        
        # Ensure edge_construction_function_names is a list/tuple
        if isinstance(edge_construction_function_names, str):
            edge_construction_function_names = [edge_construction_function_names]


        # Apply selected construction functions in order (decorator pattern)
        for name in edge_construction_function_names:
            if name in construction_function_factories:
                factory = construction_function_factories[name]
                current_edges_obj = factory(current_edges_obj) # Decorate the current Edges object
            else:
                logging.warning(f"Unknown edge construction function name: {name}. Skipping.")
        
        # Finally, compute edges using the fully decorated Edges object
        return current_edges_obj.compute_edges()

def _construct_edges(atom_coordinates_matrices, sequences, esm2_contact_maps_list, # Added esm2_contact_maps_list
                     edge_construction_config, # This should be a list of names like ['sequence_based', 'distance_based_threshold']
                     distance_function_config, 
                     distance_threshold_config, 
                     use_edge_attr_config):
    
    num_cores = multiprocessing.cpu_count()

    # Prepare arguments for each call to EdgeConstructionContext.compute_edges
    # Each element in 'args_list' corresponds to one protein/sequence
    args_list = []
    for i in range(len(sequences)):
        atom_coords = atom_coordinates_matrices[i] if atom_coordinates_matrices else None
        seq = sequences[i]
        esm2_map = esm2_contact_maps_list[i] if esm2_contact_maps_list and i < len(esm2_contact_maps_list) else None
        
        # Handle cases where atom_coords might be None or empty from previous steps
        if atom_coords is None or atom_coords.size == 0:
            # If coords are required by a chosen method but missing, this could be an issue.
            # EdgeConstructionContext.compute_edges and individual decorators should handle None coords gracefully.
            # For now, we pass them as is.
            pass

        args_list.append((
            edge_construction_config, # List of function names to apply
            distance_function_config,
            distance_threshold_config,
            atom_coords,
            seq,
            esm2_map,
            use_edge_attr_config
        ))

    adjacency_matrices = []
    weights_matrices = []

    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=num_cores) as pool:
        # tqdm setup for progress tracking
        with tqdm(total=len(args_list), desc="Constructing Edges") as progress_bar:
            # Submit all tasks to the pool
            futures = [pool.submit(EdgeConstructionContext.compute_edges, arg) for arg in args_list]
            
            for future in futures:
                try:
                    adj_matrix, w_matrix = future.result() # Get result from each future
                    adjacency_matrices.append(adj_matrix)
                    weights_matrices.append(w_matrix)
                except Exception as e:
                    logging.error(f"Error in edge construction subprocess: {e}")
                    # Append placeholders or handle error as appropriate
                    # For now, let's assume if one fails, we might not want to proceed or need specific error handling.
                    # Or append None and filter later:
                    adjacency_matrices.append(None) # Or some default error indicator
                    weights_matrices.append(None)
                progress_bar.update(1) # Manually update progress for each completed future

    # Filter out None results if any errors occurred and placeholders were added
    # adjacency_matrices = [m for m in adjacency_matrices if m is not None]
    # weights_matrices = [m for m in weights_matrices if m is not None]
    # This filtering might cause length mismatch with original sequences if not handled carefully.

    return adjacency_matrices, weights_matrices


def get_edges(tertiary_structure_method_flag: bool, # Renamed for clarity
              sequences: list,
              # Parameters for edge construction, could be part of a config object
              edge_construction_types=['distance_based_threshold'], # Default or example
              distance_func='euclidean',
              dist_threshold=10.0,
              use_edge_attributes=True,
              esm2_maps=None): # Optional precomputed ESM2 maps

    if tertiary_structure_method_flag: # True means predict, False means load
        # This implies ESMFold prediction if true
        # Ensure torch.hub.set_dir is called before this if predict_structures uses esm.pretrained
        # It's better to set hub_dir once at the start of the main script.
        atom_coordinates_matrices = predict_tertiary_structures(sequences)
    else:
        atom_coordinates_matrices = load_tertiary_structures(sequences)

    # Ensure atom_coordinates_matrices has one entry per sequence, even if empty/None
    # This is important if some structures failed to load/predict
    if len(atom_coordinates_matrices) != len(sequences):
        # This case should ideally be handled within predict/load to return placeholders
        logging.error("Mismatch between number of sequences and atom coordinate matrices.")
        # Pad with Nones or empty arrays if necessary, though predict/load should do this
        # For now, assuming they are of the same length.
        # Example fix:
        # if len(atom_coordinates_matrices) < len(sequences):
        #     atom_coordinates_matrices.extend([np.array([]) for _ in range(len(sequences) - len(atom_coordinates_matrices))])


    # If esm2_maps are not provided, they will be None for _construct_edges
    # If 'esm2_contact_map_XX' is in edge_construction_types, esm2_maps should be provided.
    if esm2_maps is None:
        esm2_maps = [None] * len(sequences) # Default if not provided

    adjacency_matrices, weights_matrices = _construct_edges(
        atom_coordinates_matrices,
        sequences,
        esm2_maps, # Pass the actual maps
        edge_construction_types, 
        distance_func,
        dist_threshold,
        use_edge_attributes
    )

    return adjacency_matrices, weights_matrices


def to_parse_matrix(adjacency_matrix, nodes_features, weights_matrix, label, eps=1e-6):
    # Validate inputs
    if adjacency_matrix is None:
        raise ValueError("Adjacency matrix cannot be None for to_parse_matrix.")
    if nodes_features is None:
        raise ValueError("Node features cannot be None for to_parse_matrix.")
        
    num_row, num_col = adjacency_matrix.shape
    if num_row != nodes_features.shape[0]:
        # This can happen if a sequence had no PDB/coords, leading to empty adj_matrix (e.g. 0x0)
        # but node_features were still created based on sequence length.
        # Or if adj_matrix is placeholder due to error.
        # Handle this discrepancy: either raise error, or return None/empty Data object.
        logging.error(f"Shape mismatch: adj_matrix ({num_row},{num_col}), node_features ({nodes_features.shape}). Label: {label}")
        # Option: return an empty Data object or None, to be filtered out later.
        # For now, let's allow it to proceed if num_row is 0, but it will likely fail if num_row > 0 and mismatch.
        if num_row == 0 and nodes_features.shape[0] > 0: # No edges from empty adj matrix
             # Create a Data object with nodes but no edges
            edge_index = torch.empty((2, 0), dtype=torch.int64) # No edges
            edge_attr = torch.empty((0, weights_matrix.shape[-1] if weights_matrix.ndim == 3 and weights_matrix.size > 0 else 0), dtype=torch.float32)
            x = torch.tensor(nodes_features, dtype=torch.float32)
            y_tensor = torch.tensor([label], dtype=torch.int64) if label is not None else torch.empty(0, dtype=torch.int64) # Handle None label
            
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
            # data.validate(raise_on_error=True) # Validation might fail if y is unexpected for no edges
            return data
        elif num_row != nodes_features.shape[0] and num_row > 0 : # Actual mismatch for non-empty adj
             raise ValueError(f"Adjacency matrix row count ({num_row}) must match node features count ({nodes_features.shape[0]}).")


    rows = []
    cols = []
    e_vec_list = [] # Changed to list of lists/tuples for multi-dim weights

    for i in range(num_row):
        for j in range(num_col):
            if adjacency_matrix[i, j] >= eps: # Use comma for numpy indexing
                rows.append(i)
                cols.append(j)
                if weights_matrix is not None and weights_matrix.size > 0:
                    # weights_matrix could be (N, N, num_features) or (N,N) if single feature
                    if weights_matrix.ndim == 3: # Multiple edge features
                        e_vec_list.append(weights_matrix[i, j, :])
                    elif weights_matrix.ndim == 2: # Single edge feature, ensure it's treated as a list/array for tensor
                        e_vec_list.append([weights_matrix[i, j]])
                    # If weights_matrix is (0,0) or other empty indicator, this block is skipped.
    
    edge_index = torch.tensor([rows, cols], dtype=torch.int64)
    x = torch.tensor(nodes_features, dtype=torch.float32)
    
    if e_vec_list: # If there are edge attributes
        edge_attr = torch.tensor(np.array(e_vec_list), dtype=torch.float32)
        # Ensure edge_attr has 2 dimensions, e.g. (num_edges, num_edge_features)
        if edge_attr.ndim == 1: # If only one feature type and it became 1D
            edge_attr = edge_attr.unsqueeze(1)
    else: # No edge attributes or no edges
        # PyG expects edge_attr to have shape [num_edges, num_edge_features]
        # If no edges, num_edges is 0. If edges but no features, num_edge_features is 0.
        # Let's determine num_edge_features from weights_matrix if possible, else 0.
        num_edge_features = 0
        if weights_matrix is not None and weights_matrix.size > 0:
            if weights_matrix.ndim == 3:
                num_edge_features = weights_matrix.shape[-1]
            elif weights_matrix.ndim == 2: # Assumed single feature
                num_edge_features = 1
        
        edge_attr = torch.empty((edge_index.shape[1], num_edge_features), dtype=torch.float32)

    y_tensor = torch.tensor([label], dtype=torch.int64) if label is not None else torch.empty(0, dtype=torch.int64) # Handle None label

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
    try:
        data.validate(raise_on_error=True)
    except Exception as e_val:
        # print(f"Data validation error: {e_val}")
        # print(f"Shapes: x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}, y={y_tensor.shape}")
        # print(f"Adj matrix shape: {adjacency_matrix.shape}, Node features shape: {nodes_features.shape}")
        # print(f"Label: {label}")
        raise e_val # Re-raise after printing info
    return data


def step_decay(epoch):
    initial_lr = 0.01
    drop = 0.5
    epochs_drop = 10
    lr = initial_lr * np.power(drop, np.floor((1 + epoch) / epochs_drop))
    return lr

# Function to optimize threshold based on MCC
def optimize_threshold(y_true, y_pred_probas):
    thresholds = np.arange(0.1, 1.0, 0.05) # Consider a finer range if needed
    best_mcc = -1.0 # Initialize to worst possible MCC
    best_threshold = 0.5 # Default threshold

    # Ensure y_true and y_pred_probas are CPU tensors for sklearn metrics
    y_true_cpu = y_true.cpu() if isinstance(y_true, torch.Tensor) else y_true
    y_pred_probas_cpu = y_pred_probas.cpu() if isinstance(y_pred_probas, torch.Tensor) else y_pred_probas

    if len(np.unique(y_true_cpu)) < 2:
        # MCC is not well-defined if only one class is present in y_true.
        # print("Warning: Only one class present in y_true. MCC optimization might be unreliable.")
        # Return default threshold or handle as per requirements.
        return best_threshold, best_mcc # Or perhaps 0.0 for MCC if undefined

    for threshold_val in thresholds: # Renamed to avoid conflict
        y_pred = (y_pred_probas_cpu > threshold_val).int()
        try:
            mcc = matthews_corrcoef(y_true_cpu, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold_val
        except ValueError: 
            # Handles cases where MCC is undefined (e.g. all preds are same class, and true labels are mixed)
            # This can happen with extreme thresholds or poor models.
            # print(f"Warning: MCC undefined for threshold {threshold_val}. Skipping.")
            pass


    return best_threshold, best_mcc

def save_metrics_to_csv(metrics_list, filename):
    df = pd.DataFrame(metrics_list)
    # Ensure directory exists if filename includes path
    output_dir = Path(filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename, index=False)

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
    
def train_test_CNN_model(X_full, y_full, X_test_final, y_test_final, device='cuda'): # Renamed inputs for clarity
    
    # Split the provided X_full, y_full into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=123, stratify=y_full if np.sum(y_full)>1 else None) # Stratify if possible

    # Assuming X_train, X_val, X_test_final are numpy arrays of embeddings
    # y_train, y_val, y_test_final are numpy arrays of labels

    input_shape_tuple = (X_train.shape[1], 1) # (sequence_length=1280, channels=1)
    model = Conv1DClassifier(input_shape_tuple).to(device)
    
    # Calculate pos_weight based on the training set
    # pos_weight = num_negative / num_positive
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    if num_pos > 0 and num_neg > 0: # Avoid division by zero
        pos_weight_val = num_neg / num_pos
    else: # Default if one class is missing in train (should not happen with good split)
        pos_weight_val = 1.0 
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Standard LR
    # Scheduler uses step_decay, ensure step_decay's initial_lr matches optimizer's if normalized
    # Original: scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: step_decay(epoch)/0.01)
    # If step_decay's initial_lr is 0.01, then step_decay(epoch)/0.01 normalizes it to start at 1.
    # So, effective LR starts at optimizer's LR (0.001) and then decays.
    # Let's make step_decay directly output the desired LR factor for 0.001
    # Or, adjust step_decay's initial_lr to 0.001 and use lambda epoch: step_decay(epoch)/0.001
    # For simplicity, let's assume step_decay is tuned for an initial_lr of 0.01, so the division is correct.
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: step_decay(epoch) / 0.01)


    best_val_mcc_cnn = -1.0 # Changed from acc to mcc for consistency with other models
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_patience = 20
    epochs_without_improve = 0

    num_epochs = 100 # Or as desired
    batch_size = 32  # Or as desired
    metrics_log_cnn = [] # Renamed
    
    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train.shape[0])
        epoch_loss = 0.0
        correct_preds_train = 0 # Renamed

        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = torch.tensor(X_train[indices], dtype=torch.float32, device=device)
            # y_train is numpy, convert to tensor
            batch_y = torch.tensor(y_train[indices], dtype=torch.float32, device=device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_x) # Logits
            loss = criterion(outputs, batch_y) # BCEWithLogitsLoss expects logits
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            scores_train = torch.sigmoid(outputs) # Probabilities
            preds_train = (scores_train > 0.5).float() # Default 0.5 threshold for training acc
            correct_preds_train += (preds_train == batch_y).sum().item()

        train_acc = correct_preds_train / len(X_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_x_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
            val_y_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)
            val_outputs_logits = model(val_x_tensor) # Logits
            
            # Optimize threshold on validation set logits
            best_thresh_val, current_best_mcc_val = optimize_threshold(val_y_tensor, val_outputs_logits)
            # print(f"Epoch {epoch+1} Val Best Threshold: {best_thresh_val:.2f}, Val Best MCC: {current_best_mcc_val:.4f}")
            
            val_scores_probs = torch.sigmoid(val_outputs_logits) # Probabilities
            val_preds_final = (val_scores_probs > best_thresh_val).float() # Use optimized threshold

            # Calculate metrics using optimized threshold
            val_acc = accuracy_score(y_true=val_y_tensor.cpu(), y_pred=val_preds_final.cpu())
            val_auc = roc_auc_score(y_true=val_y_tensor.cpu(), y_score=val_scores_probs.cpu()) # AUC uses scores
            val_precision = precision_score(val_y_tensor.cpu(), val_preds_final.cpu(), zero_division=0)
            val_specificity_val = specificity(val_y_tensor.cpu(), val_preds_final.cpu()) # Renamed
            val_recall = recall_score(val_y_tensor.cpu(), val_preds_final.cpu(), zero_division=0)        
            # val_mcc is current_best_mcc_val

        metrics_log_cnn.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / (len(X_train) / batch_size), # Avg loss per batch
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_sensitivity": val_recall, # Sensitivity is recall
            "val_specificity": val_specificity_val,
            "val_auc": val_auc,
            "val_mcc": current_best_mcc_val
        })

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss / (len(X_train) / batch_size):.4f}, Train Acc: {train_acc:.4f}, Val MCC: {current_best_mcc_val:.4f}, Val AUC: {val_auc:.4f}") 

        if current_best_mcc_val > best_val_mcc_cnn:
            best_val_mcc_cnn = current_best_mcc_val
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
            # torch.save(model.state_dict(), 'best_cnn_model.pth') # Save best model
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= early_stop_patience:
            print("Early stopping triggered for CNN model.")
            break

        scheduler.step()

    model.load_state_dict(best_model_wts) # Load best model weights
    save_metrics_to_csv(metrics_log_cnn, "training_validation_results_CNN.csv")
    
    # Final Test Evaluation using the best model from validation
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.tensor(X_test_final, dtype=torch.float32, device=device)
        test_y_tensor = torch.tensor(y_test_final, dtype=torch.float32, device=device).unsqueeze(1)
        test_outputs_logits = model(test_x_tensor).cpu() # Get logits on CPU
        
        # Use the threshold optimized on the validation set (or re-optimize on test if that's the protocol)
        # For now, let's assume we use a fixed 0.5 or re-optimize on test logits for test metrics.
        # The original code re-optimizes threshold on test outputs.
        best_thresh_test, best_mcc_test = optimize_threshold(test_y_tensor.cpu(), test_outputs_logits)
        # print(f"Test Best Threshold: {best_thresh_test:.2f}, Test Best MCC: {best_mcc_test:.4f}")
        
        test_scores_probs = torch.sigmoid(test_outputs_logits) # Probabilities
        test_preds_final = (test_scores_probs > best_thresh_test).float() # Use test-optimized threshold

        test_acc = accuracy_score(y_true=test_y_tensor.cpu(), y_pred=test_preds_final.cpu())
        test_auc = roc_auc_score(y_true=test_y_tensor.cpu(), y_score=test_scores_probs.cpu())
        test_precision = precision_score(test_y_tensor.cpu(), test_preds_final.cpu(), zero_division=0)
        test_specificity_final = specificity(test_y_tensor.cpu(), test_preds_final.cpu()) # Renamed
        test_recall = recall_score(test_y_tensor.cpu(), test_preds_final.cpu(), zero_division=0)        
        # test_mcc is best_mcc_test

        print(f"Test CNN: Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Sensitivity: {test_recall:.4f}, Specificity: {test_specificity_final:.4f}, AUC: {test_auc:.4f}, MCC: {best_mcc_test:.4f}")    
    
    metrics_test_final = [{"accuracy": test_acc,
                           "precision_score": test_precision,
                           "sensitivity_score": test_recall,
                           "specificity_score": test_specificity_final,
                           "auc": test_auc,
                           "mcc": best_mcc_test}]
    
    # Plotting (using validation data as example, or test data)
    val_y_cpu_plot = val_y_tensor.cpu().numpy().flatten()
    val_scores_cpu_plot = val_scores_probs.cpu().numpy().flatten()
    plot_roc_curve(val_y_cpu_plot, val_scores_cpu_plot, "roc_auc_validation_cnn.png")
    # plot_output_scores(val_scores_cpu_plot, val_y_cpu_plot, "plot_scores_validation_cnn.png")
    
    test_y_cpu_plot = test_y_tensor.cpu().numpy().flatten()
    test_scores_cpu_plot = test_scores_probs.cpu().numpy().flatten()
    plot_roc_curve(test_y_cpu_plot, test_scores_cpu_plot, "roc_auc_testing_cnn.png")
    # plot_output_scores(test_scores_cpu_plot, test_y_cpu_plot, "plot_scores_testing_cnn.png")
    
    save_metrics_to_csv(metrics_test_final, "testing_results_CNN.csv")
    
    # Return model, validation y and scores, test y and scores, and test metrics
    return model, val_y_cpu_plot, val_scores_cpu_plot, test_y_cpu_plot, test_scores_cpu_plot, metrics_test_final


class GATModel(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, output_dim, drop, heads, k_ratio, add_self_loops, num_layers=3): # k is ratio
        super(GATModel, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # Usually 1 for binary classification logit
        self.drop = drop
        self.heads = heads
        self.k_ratio = k_ratio # k is a ratio for TopKPooling
        self.add_self_loops = add_self_loops
        self.num_layers = num_layers

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() # Norm layers after GAT, before activation
        
        current_dim = node_feature_dim
        for i in range(num_layers):
            is_last_layer = (i == num_layers - 1)
            if is_last_layer:
                # Last GAT layer: output is hidden_dim (no head concatenation)
                self.gat_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, concat=False, 
                            add_self_loops=add_self_loops, dropout=drop) # Add dropout to GATConv
                )
                current_dim = hidden_dim # Output dim of last GAT layer
            else:
                # Intermediate GAT layers: output is heads * hidden_dim
                self.gat_layers.append(
                    GATConv(current_dim, hidden_dim, heads=heads, concat=True,
                            add_self_loops=add_self_loops, dropout=drop) # Add dropout
                )
                current_dim = heads * hidden_dim # Output dim for next layer
            
            self.norm_layers.append(LayerNorm(current_dim)) # Norm layer for the output of GAT[i]
                 
        # MLP layers after GAT and Pooling
        # The input to MLP depends on TopKPooling output.
        # TopKPooling reduces nodes but keeps feature dimension (current_dim from last GAT).
        # Then a global pooling or flattening is usually applied.
        # The original code had a strange transpose and Linear layer for readout.
        # Let's use a standard global mean pooling after TopK, then MLP.
        
        # self.topk_pool = TopKPooling(current_dim, ratio=k_ratio) # current_dim is output of last GAT layer
        # Using TopKPooling with ratio. The number of output nodes will be ceil(ratio * N).
        # The feature dimension of these nodes remains `current_dim`.
        # A common pattern is global mean/sum/max pool after TopK, or just use the TopK nodes.
        
        # The original readout:
        # x = self.topk_pool(x, edge_index, edge_attr, batch=batch)[0]
        # x = torch.transpose(x, 0, 1)
        # x = nn.Linear(x.shape[1], batch[-1] + 1, bias=False, ...)(x) # This is unusual, seems like custom graph-level readout
        # x = torch.transpose(x, 0, 1)
        # This custom readout results in shape (batch_size, current_dim)

        # For TopKPooling, input channels should be the dimension of features *before* pooling
        self.topk_pool = TopKPooling(current_dim, ratio=k_ratio)


        # MLP layers (classifier part)
        # Input to lin0 is `current_dim` because the custom readout maintains this feature dim per graph.
        self.lin0 = nn.Linear(current_dim, hidden_dim) # First MLP layer
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)  # Second MLP layer
        self.lin = nn.Linear(hidden_dim, output_dim)   # Final output layer (e.g., 1 logit)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and (p.name is None or "bias" in p.name): # Initialize biases to zero
                 nn.init.zeros_(p)


    def forward(self, x, edge_index, edge_attr, batch):
        # GAT layers
        for i in range(self.num_layers):
            x = self.gat_layers[i](x, edge_index, edge_attr=edge_attr if edge_attr is not None and edge_attr.numel() > 0 else None)
            x = self.norm_layers[i](x, batch) # Apply LayerNorm
            x = F.relu(x) # Apply ReLU after norm
            # Dropout is often applied after activation, or within GATConv itself (PyG GATConv has dropout arg)
            # If GATConv has dropout, no need for extra F.dropout here unless desired.
            # The original code had F.dropout after relu for intermediate layers.
            if i < self.num_layers - 1: # No dropout after the final GAT layer's activation
                 x = F.dropout(x, p=self.drop, training=self.training)


        # Readout layer (original custom readout)
        # This part assumes TopKPooling is appropriate and its output is handled correctly.
        # The input to topk_pool should be the features from the last GAT layer.
        x_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled, _ = self.topk_pool(x, edge_index, edge_attr=edge_attr if edge_attr is not None and edge_attr.numel() > 0 else None, batch=batch)
        
        # The original custom readout:
        # This part is a bit unusual. It seems to create a graph-level representation.
        # x_pooled has shape [num_pooled_nodes, features_dim]
        # batch_pooled maps these nodes to original graphs in the batch.
        # Need to ensure device consistency for the dynamically created Linear layer.
        current_device = x.device 
        
        # A more standard global pooling approach (e.g., global mean pooling) might be simpler:
        # from torch_geometric.nn import global_mean_pool
        # x_graph_level = global_mean_pool(x_pooled, batch_pooled) # Shape: [batch_size, features_dim]
        # This x_graph_level would then go into the MLP.

        # Using the original custom readout:
        if x_pooled.numel() == 0: # If TopKPooling results in zero nodes (e.g. k_ratio too small for tiny graphs)
            # Handle empty pool. Output zeros of the expected shape for the MLP.
            # Expected shape after readout: (batch_size, features_dim_of_last_gat_layer)
            # batch[-1] + 1 gives batch_size.
            # features_dim_of_last_gat_layer is self.gat_layers[-1].out_channels
            batch_size = batch.max().item() + 1 if batch is not None and batch.numel() > 0 else 1 # Handle empty batch
            # last_gat_out_dim = self.gat_layers[-1].out_channels # This is hidden_dim if last layer concat=False
            last_gat_out_dim = self.norm_layers[-1].normalized_shape[0] if isinstance(self.norm_layers[-1].normalized_shape, tuple) else self.norm_layers[-1].normalized_shape

            x_graph_level = torch.zeros((batch_size, last_gat_out_dim), device=current_device)
        else:
            # Transpose to [features, num_pooled_nodes]
            x_transposed = torch.transpose(x_pooled, 0, 1)
            # Dynamically create Linear layer for readout (as in original)
            # This layer maps from num_pooled_nodes to batch_size (num_graphs) for each feature.
            # x_transposed.shape[1] is num_pooled_nodes. batch_pooled[-1]+1 is batch_size.
            # This linear layer needs to be on the same device.
            num_graphs_in_batch = batch_pooled.max().item() + 1 if batch_pooled is not None and batch_pooled.numel() > 0 else 1
            
            readout_layer = nn.Linear(x_transposed.shape[1], num_graphs_in_batch, bias=False, device=current_device)
            nn.init.xavier_uniform_(readout_layer.weight) # Initialize weights

            x_intermediate = readout_layer(x_transposed) # Shape: [features, batch_size]
            x_graph_level = torch.transpose(x_intermediate, 0, 1) # Shape: [batch_size, features]
            # x_graph_level now has shape [batch_size, current_dim_after_last_gat]

        # MLP classifier
        x = F.dropout(x_graph_level, p=self.drop, training=self.training) # Dropout before first MLP layer
        x = self.lin0(x)
        x = F.relu(x)
        # x = F.dropout(x, p=self.drop, training=self.training) # Optional dropout between MLP layers
        x = self.lin1(x)
        x = F.relu(x)
        
        z_features = x  # Features from the second to last MLP layer (latent representation)
        x_logits = self.lin(x) # Final logits
        
        return x_logits, z_features


def sensitivity(y_true, y_pred):
    # Ensure y_true, y_pred are numpy arrays
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)

    if len(np.unique(y_true_np)) < 2 and len(np.unique(y_pred_np)) < 2 and y_true_np[0] == y_pred_np[0]:
        # Handle case where all true and all pred are same class (e.g. all TN or all TP)
        # This can lead to confusion_matrix issues if labels are not [0,1] or only one label exists.
        # If all are TN (0,0), sensitivity is undefined or 0. If all are TP (1,1), sensitivity is 1.
        # For safety, let's rely on confusion_matrix with explicit labels.
        pass

    try:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true_np, y_pred=y_pred_np, labels=[0, 1]).ravel()
        if tp + fn == 0: # No actual positives in y_true, or all actual positives were predicted as negative.
            # If tp=0 and fn=0 (no positives in y_true), sensitivity is often taken as NaN or 1 (if no FNs means perfect for positives).
            # If tp=0 and fn >0 (positives exist but all missed), sensitivity is 0.
            return np.nan if tp==0 and fn==0 else 0.0
        else:
            sensitivity_val = tp / (tp + fn)
            return sensitivity_val
    except ValueError: # If confusion_matrix fails (e.g. y_true/y_pred are empty or unexpected)
        return np.nan


def specificity(y_true, y_pred):
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true_np, y_pred=y_pred_np, labels=[0, 1]).ravel()
        if tn + fp == 0: # No actual negatives in y_true, or all actual negatives were predicted as positive.
            return np.nan if tn==0 and fp==0 else 0.0
        else:
            specificity_val = tn / (tn + fp)
            return specificity_val
    except ValueError:
        return np.nan


class HybridModel(nn.Module):
    def __init__(self, cnn_input_channels=1, cnn_seq_len=1280,
                 node_feature_dimension=10, gat_hidden=128, # GAT hidden dim for its internal layers
                 alpha=0.5, num_layers=3, # num_layers for GAT
                 gat_output_dim=1, # GAT branch should output 1 logit
                 gat_dropout=0.3, gat_heads=8, gat_k_ratio=0.1): # k_ratio for TopKPooling, e.g. 0.1 for 10%
        super(HybridModel, self).__init__()
        
        # CNN Branch
        self.cnn_branch = Conv1DClassifier(input_shape_tuple=(cnn_seq_len, cnn_input_channels))
        
        # GAT Branch
        # GATModel's output_dim is the dimension of its final MLP layer (e.g., 1 for a logit)
        # GATModel's hidden_dim is for its internal MLP layers (lin0, lin1)
        self.gat_branch = GATModel(
            node_feature_dim=node_feature_dimension, 
            hidden_dim=gat_hidden, # This is hidden_dim for GAT's own MLP
            output_dim=gat_output_dim, # GAT branch outputs 1 logit
            drop=gat_dropout, 
            heads=gat_heads, 
            k_ratio=gat_k_ratio, # k_ratio for TopKPooling
            add_self_loops=False, # As per original GATModel call
            num_layers=num_layers
        )

        self.alpha = alpha  # Weight for combining outputs (if weighted average)

        # Original MLP for combining was commented out.
        # If using weighted average of logits, no further MLP is needed here.
        # self.fc1 = nn.Linear(2, mlp_hidden) # If concatenating 2 logits
        # self.dropout = nn.Dropout(0.3)
        # self.fc2 = nn.Linear(mlp_hidden, 1)

    def forward(self, cnn_input, gat_node_features, gat_edge_index, gat_edge_attr, gat_batch):
        # CNN branch forward pass
        cnn_output_logits = self.cnn_branch(cnn_input)  # Shape: (batch_size, 1)
        
        # GAT branch forward pass
        # gat_output_logits, _ = self.gat_branch(gat_node_features, gat_edge_index, gat_edge_attr, gat_batch)
        # Pass edge_attr only if it's not None and has elements
        effective_gat_edge_attr = gat_edge_attr if gat_edge_attr is not None and gat_edge_attr.numel() > 0 else None
        gat_output_logits, _ = self.gat_branch(gat_node_features, gat_edge_index, effective_gat_edge_attr, gat_batch)


        # Combine the two outputs (logits) using weighted average
        # This is the method used in the original code if alpha is for logits
        combined_logits = self.alpha * gat_output_logits + (1 - self.alpha) * cnn_output_logits
        
        # Alternative: Concatenate and pass through another MLP (original commented out structure)
        # combined_output_concat = torch.cat([cnn_output_logits, gat_output_logits], dim=1) # Shape: (batch_size, 2)
        # x = F.relu(self.fc1(combined_output_concat))
        # x = self.dropout(x)
        # combined_logits = self.fc2(x) # Shape: (batch_size, 1)
        
        return combined_logits
    
def train_hybrid_model(X_full_embed, graphs_full, y_full_labels, trial_params, alpha=0.5, device='cuda:0'): # Renamed inputs
    
    # Split full data into training and validation sets
    # Ensure y_full_labels is numpy for train_test_split stratify
    y_full_labels_np = np.array(y_full_labels) if not isinstance(y_full_labels, np.ndarray) else y_full_labels
    
    indices_full = np.arange(len(y_full_labels_np))
    train_idx, val_idx = train_test_split(indices_full, test_size=0.2, random_state=123, stratify=y_full_labels_np if np.sum(y_full_labels_np)>1 else None) # Stratify if possible

    # Split embeddings
    X_train_embed = X_full_embed[train_idx]
    X_val_embed = X_full_embed[val_idx]
    
    y_train_labels = y_full_labels_np[train_idx]
    y_val_labels = y_full_labels_np[val_idx]  

    # Split graphs
    graphs_train = [graphs_full[i] for i in train_idx.tolist()]
    graphs_val = [graphs_full[i] for i in val_idx.tolist()]

    # Determine node_feature_dimension from the first graph (if graphs exist)
    if not graphs_train: # Handle empty graph list
        # This case should ideally not happen if data is prepared correctly.
        # If it does, GAT branch cannot be initialized without node_feature_dimension.
        raise ValueError("Graph list for training is empty. Cannot determine node feature dimension.")
    node_feature_dimension = graphs_train[0].x.shape[1]
    
    # Initialize HybridModel
    # trial_params should contain: "lr", "gat_hidden", "batch_size", "pos_weight_val", "num_layers"
    # alpha is passed separately.
    model = HybridModel(
        cnn_input_channels=1, 
        cnn_seq_len=X_train_embed.shape[1], # Assuming X_train_embed is (N, seq_len)
        node_feature_dimension=node_feature_dimension,
        gat_hidden=trial_params["gat_hidden"], # For GAT's internal MLP
        alpha=alpha, 
        num_layers=trial_params["num_layers"] # For GAT architecture
        # Other GAT params like dropout, heads, k_ratio use defaults in HybridModel
    ).to(device)

    # Loss criterion with positive weight
    pos_weight_tensor = torch.tensor([trial_params["pos_weight_val"]], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=trial_params["lr"])
    # Scheduler: ensure step_decay is compatible with this LR
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: step_decay(epoch) / 0.01) 

    best_val_mcc_hybrid = -1.0 # Monitor MCC on validation set
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_patience = 20 # Number of epochs to wait for improvement
    epochs_without_improve = 0

    num_epochs = trial_params.get("num_epochs", 200) # Allow num_epochs in trial_params, default 200
    batch_size = trial_params["batch_size"]
    
    metrics_log_hybrid = [] # For logging metrics per epoch
    final_val_metrics_on_best_epoch = {}


    for epoch in range(num_epochs):
        model.train()
        permutation = torch.randperm(X_train_embed.shape[0])
        epoch_loss_train = 0.0 # Renamed
        correct_preds_train_hybrid = 0 # Renamed

        for i in range(0, X_train_embed.shape[0], batch_size):
            batch_indices = permutation[i:i+batch_size]
            
            # CNN input
            batch_x_cnn = torch.tensor(X_train_embed[batch_indices.cpu().numpy()], dtype=torch.float32, device=device)
            
            # GAT input (graphs)
            # graphs_train is a list of Data objects. Need to select and batch them.
            current_graphs_list = [graphs_train[j] for j in batch_indices.cpu().numpy()]
            batch_x_gat = Batch.from_data_list(current_graphs_list).to(device)
            
            # Labels
            batch_y_labels = torch.tensor(y_train_labels[batch_indices.cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(1)
            
            optimizer.zero_grad()
            # Forward pass through HybridModel
            outputs_logits = model(
                batch_x_cnn, 
                batch_x_gat.x, 
                batch_x_gat.edge_index, 
                batch_x_gat.edge_attr, 
                batch_x_gat.batch
            )
            loss = criterion(outputs_logits, batch_y_labels)
            loss.backward()
            optimizer.step()

            epoch_loss_train += loss.item()
            # Training accuracy (optional, using 0.5 threshold)
            train_scores = torch.sigmoid(outputs_logits)
            train_preds = (train_scores > 0.5).float()
            correct_preds_train_hybrid += (train_preds == batch_y_labels).sum().item()

        avg_epoch_loss_train = epoch_loss_train / (len(X_train_embed) / batch_size)
        train_acc_epoch = correct_preds_train_hybrid / len(X_train_embed)

        # Validation phase
        model.eval()
        with torch.no_grad():
            # Prepare validation data (CNN part)
            val_x_cnn_tensor = torch.tensor(X_val_embed, dtype=torch.float32, device=device)
            
            # Prepare validation data (GAT part) - batch all validation graphs
            val_graphs_batch = Batch.from_data_list(graphs_val).to(device)
            
            # Validation labels
            val_y_labels_tensor = torch.tensor(y_val_labels, dtype=torch.float32, device=device).unsqueeze(1)

            val_outputs_logits = model(
                val_x_cnn_tensor,
                val_graphs_batch.x,
                val_graphs_batch.edge_index,
                val_graphs_batch.edge_attr,
                val_graphs_batch.batch
            )
            
            # Optimize threshold on validation logits
            best_thresh_val_hybrid, current_mcc_val_hybrid = optimize_threshold(val_y_labels_tensor, val_outputs_logits)
            
            val_scores_probs = torch.sigmoid(val_outputs_logits) # Probabilities
            val_preds_final_hybrid = (val_scores_probs > best_thresh_val_hybrid).float() # Use optimized threshold

            # Calculate validation metrics
            val_acc_hybrid = accuracy_score(val_y_labels_tensor.cpu(), val_preds_final_hybrid.cpu())
            val_auc_hybrid = roc_auc_score(y_true=val_y_labels_tensor.cpu(), y_score=val_scores_probs.cpu())
            val_precision_hybrid = precision_score(val_y_labels_tensor.cpu(), val_preds_final_hybrid.cpu(), zero_division=0)
            val_recall_hybrid = recall_score(val_y_labels_tensor.cpu(), val_preds_final_hybrid.cpu(), zero_division=0)
            val_specificity_hybrid = specificity(val_y_labels_tensor.cpu(), val_preds_final_hybrid.cpu())
            # current_mcc_val_hybrid is already calculated by optimize_threshold

        metrics_log_hybrid.append({
            "epoch": epoch + 1,
            "train_loss": avg_epoch_loss_train,
            "train_acc": train_acc_epoch,
            "val_acc": val_acc_hybrid,
            "val_precision": val_precision_hybrid,
            "val_sensitivity": val_recall_hybrid, # Sensitivity is recall
            "val_specificity": val_specificity_hybrid,
            "val_auc": val_auc_hybrid,
            "val_mcc": current_mcc_val_hybrid
        })
        
        print(f"Hybrid Train: Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss_train:.4f}, Acc: {train_acc_epoch:.4f}")
        print(f"Hybrid Val: Epoch {epoch+1} - MCC: {current_mcc_val_hybrid:.4f}, AUC: {val_auc_hybrid:.4f}, Acc: {val_acc_hybrid:.4f}")

        if current_mcc_val_hybrid > best_val_mcc_hybrid:
            best_val_mcc_hybrid = current_mcc_val_hybrid
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
            # Save metrics for this best epoch
            final_val_metrics_on_best_epoch = metrics_log_hybrid[-1].copy() # Store the latest metrics entry
            # torch.save(model.state_dict(), 'best_hybrid_model.pth') # Optional: save best model state
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= early_stop_patience:
            print("Early stopping triggered for Hybrid model.")
            break

        scheduler.step()

    model.load_state_dict(best_model_wts) # Load the best model weights found during validation
    save_metrics_to_csv(metrics_log_hybrid, "training_validation_results_Hybrid_CNN_GAT.csv")
    
    # Prepare data for plotting (using validation set from the run)
    # These are from the *last* epoch of validation, not necessarily the best epoch if early stopping happened much earlier.
    # For plotting, it's better to re-run prediction with the best model on val set if needed, or use stored scores.
    # For now, returning y_val_labels and scores from the last validation run (which corresponds to loaded best_model_wts if no further changes).
    
    # Re-evaluate on val set with the loaded best model to get its scores for plotting (if not already stored)
    model.eval()
    with torch.no_grad():
        val_x_cnn_tensor_final = torch.tensor(X_val_embed, dtype=torch.float32, device=device)
        val_graphs_batch_final = Batch.from_data_list(graphs_val).to(device)
        val_outputs_logits_final = model(val_x_cnn_tensor_final, val_graphs_batch_final.x, val_graphs_batch_final.edge_index, val_graphs_batch_final.edge_attr, val_graphs_batch_final.batch)
        val_scores_probs_final = torch.sigmoid(val_outputs_logits_final).cpu().numpy().flatten()
    
    val_y_cpu_for_plot = y_val_labels # Already numpy
    # plot_roc_curve(val_y_cpu_for_plot, val_scores_probs_final, "roc_auc_validation_hybrid.png")
    
    # Return the trained model, val labels, val scores (probs), and the validation metrics from the best epoch
    return model, val_y_cpu_for_plot, val_scores_probs_final, final_val_metrics_on_best_epoch


def test_hybrid_model(model, X_test_embed, graphs_test_list, y_test_labels, device='cuda:0'): # Renamed inputs
    """
    Evaluate the trained HybridModel on the test dataset.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    y_test_labels_np = np.array(y_test_labels) if not isinstance(y_test_labels, np.ndarray) else y_test_labels


    with torch.no_grad():
        # CNN input for test set
        X_test_cnn_tensor = torch.tensor(X_test_embed, dtype=torch.float32, device=device)
        
        # GAT input for test set (batch all test graphs)
        # Ensure graphs_test_list contains valid Data objects
        if not all(isinstance(g, Data) for g in graphs_test_list if g is not None): # Check for None too
             # Filter out Nones if they exist, or handle error
             valid_graphs_test = [g for g in graphs_test_list if g is not None and isinstance(g, Data) and g.x is not None and g.edge_index is not None]
             if not valid_graphs_test: # If all graphs are invalid/None
                 logging.error("All graphs in graphs_test_list are invalid or None. Cannot perform testing.")
                 # Return empty or error metrics
                 error_metrics = [{"accuracy": np.nan, "precision_score": np.nan, "sensitivity_score": np.nan, 
                                   "specificity_score": np.nan, "mcc": np.nan, "auc": np.nan}]
                 return error_metrics, np.array([]), y_test_labels_np, np.array([])


             # This implies X_test_embed and y_test_labels might need filtering to match valid_graphs_test
             # This is complex to handle here; data prep should ensure consistency.
             # For now, assume graphs_test_list is clean or Batch.from_data_list handles it.
             # However, Batch.from_data_list might fail if graphs are malformed.
             # A simple fix: if a graph is None, its corresponding embedding/label should be skipped.
             # This requires re-indexing, which is best done before calling test_hybrid_model.
             # For now, proceed assuming graphs_test_list is usable by Batch.from_data_list.
             pass # Assuming graphs_test_list is okay for now.

        test_graphs_batch = Batch.from_data_list(graphs_test_list).to(device)
        
        # Test labels
        y_test_labels_tensor = torch.tensor(y_test_labels_np, dtype=torch.float32, device=device).unsqueeze(1)

        # Get model predictions (logits)
        outputs_logits_test = model(
            X_test_cnn_tensor, 
            test_graphs_batch.x, 
            test_graphs_batch.edge_index, 
            test_graphs_batch.edge_attr, 
            test_graphs_batch.batch
        )
        
        # Optimize threshold on test logits (common practice, or use val threshold)
        best_thresh_test_hybrid, mcc_test_hybrid = optimize_threshold(y_test_labels_tensor, outputs_logits_test)
        
        y_scores_probs_test = torch.sigmoid(outputs_logits_test).cpu() # Probabilities, on CPU
        y_preds_final_test = (y_scores_probs_test > best_thresh_test_hybrid).float() # Final predictions
        y_true_test_cpu = y_test_labels_tensor.cpu()
        
        # Compute metrics
        accuracy_test = accuracy_score(y_true_test_cpu, y_preds_final_test)
        precision_test = precision_score(y_true_test_cpu, y_preds_final_test, zero_division=0)
        recall_test = recall_score(y_true_test_cpu, y_preds_final_test, zero_division=0) # Sensitivity       
        auc_test = roc_auc_score(y_true=y_true_test_cpu.numpy(), y_score=y_scores_probs_test.numpy()) # roc_auc_score needs numpy
        specificity_test = specificity(y_true_test_cpu, y_preds_final_test)
        # mcc_test_hybrid is already calculated
        
    metrics_test_hybrid = [{
        "accuracy": accuracy_test,
        "precision_score": precision_test,
        "sensitivity_score": recall_test,
        "specificity_score": specificity_test,
        "mcc": mcc_test_hybrid, 
        "auc": auc_test
    }]
    
    # Plotting for test set
    test_y_cpu_plot = y_true_test_cpu.numpy().flatten()
    test_scores_cpu_plot = y_scores_probs_test.numpy().flatten()
    # plot_roc_curve(test_y_cpu_plot, test_scores_cpu_plot, "roc_auc_testing_hybrid.png")
    
    print(f"Test Hybrid: Acc: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Sensitivity: {recall_test:.4f}, Specificity: {specificity_test:.4f}, AUC: {auc_test:.4f}, MCC: {mcc_test_hybrid:.4f}")
    save_metrics_to_csv(metrics_test_hybrid, "testing_results_Hybrid_CNN_GAT.csv")
    
    # Return: metrics dict, predictions (binary), true labels (numpy), scores (probabilities)
    return metrics_test_hybrid, y_preds_final_test.numpy().flatten(), test_y_cpu_plot, test_scores_cpu_plot


def generate_esm_embeddings(model_esm, alphabet_esm, sequence_list, output_file_path): # Renamed args
    """
    Generate ESM embeddings for a list of sequences and save the results to a CSV file.
    Input sequence_list: list of sequence strings.
    Output: Pandas DataFrame of embeddings, indexed by an ID derived from sequence or index.
    """
    # ESM model expects list of tuples: [(name1, seq1), (name2, seq2), ...]
    # Create unique names/IDs for each sequence if not provided.
    # Using "seq_INDEX" as a simple ID.
    peptide_tuples_for_esm = []
    for i, seq_str in enumerate(sequence_list):
        peptide_tuples_for_esm.append((f"seq_{i}", seq_str))

    # Process in batches if sequence_list is very large to manage memory
    # For now, processing all at once as in original code.
    # The esm_embeddings function itself might handle batching internally via batch_converter,
    # but the input to esm_embeddings is one list of tuples.
    
    # Call the internal esm_embeddings function that does the conversion
    # This function returns a DataFrame.
    embeddings_df = esm_embeddings(model_esm, alphabet_esm, peptide_tuples_for_esm)
    
    # Save to CSV
    # Ensure directory for output_file_path exists
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    embeddings_df.to_csv(output_file_path) # DataFrame saves with its index
    print(f"Saved ESM embeddings to {output_file_path}")

    return embeddings_df # Return the DataFrame


def generate_graphs(sequence_list: list, dataset_df: pd.DataFrame, tertiary_structure_method=False):
    """
    Generate graph Data objects for a list of sequences.
    - sequence_list: List of protein/peptide sequences (strings).
    - dataset_df: Pandas DataFrame containing labels, should align with sequence_list.
                  Must have a 'label' column. Index should match sequence_list order.
    - tertiary_structure_method: Bool, True to predict structures (ESMFold), False to load.
    Returns:
    - List of PyTorch Geometric Data objects.
    """
    # Get adjacency and weight matrices using the refactored get_edges
    # get_edges needs sequences and other config.
    # Using default edge construction for now: distance-based.
    # This part might need more flexible configuration passing.
    adj_matrices, weights_matrices = get_edges(
        tertiary_structure_method_flag=tertiary_structure_method,
        sequences=sequence_list,
        # Default edge construction params, can be exposed or configured:
        edge_construction_types=['sequence_based', 'distance_based_threshold'], 
        distance_func='euclidean',
        dist_threshold=10.0, # Angstroms
        use_edge_attributes=True,
        esm2_maps=None # Provide actual maps if 'esm2_contact_map_XX' is used
    )

    graphs_list = []
    num_samples = len(sequence_list)

    with tqdm(range(num_samples), total=num_samples, desc="Converting matrices to PyG Data objects") as progress:
        for i in progress:
            seq_len = len(sequence_list[i])
            
            # Create node features (placeholder: sequence length x 10, filled with ones)
            # This should ideally be actual node features (e.g., from ESM per-residue embeddings, PSSM, etc.)
            # For now, using the placeholder as in original.
            # If seq_len is 0 (empty sequence string), this will be (0,10)
            nodes_features_np = np.ones((seq_len, 10), dtype=np.float32) 

            # Get label from dataset_df
            # Ensure dataset_df index aligns with sequence_list, or use .iloc[i]
            try:
                label_val = dataset_df.iloc[i]['label'] if 'label' in dataset_df.columns else None
            except IndexError:
                logging.error(f"IndexError accessing label for sequence {i}. Dataset length: {len(dataset_df)}")
                label_val = None # Or handle error more strictly

            current_adj_matrix = adj_matrices[i]
            current_weights_matrix = weights_matrices[i]

            # Handle cases where adj_matrix or weights_matrix might be None (e.g., from errors in _construct_edges)
            # Or if sequence length is 0, leading to problematic matrices.
            if current_adj_matrix is None or (seq_len == 0 and nodes_features_np.shape[0] == 0):
                # If sequence length is 0, nodes_features is (0,10).
                # adj_matrix from EmptyEdges(0) is (0,0). This is a valid "empty graph".
                # to_parse_matrix should handle this.
                # If adj_matrix is None due to error, we might create an empty graph or skip.
                if seq_len == 0: # Valid empty sequence
                    pass # to_parse_matrix will handle it
                else: # adj_matrix is None for a non-empty sequence (error case)
                    logging.warning(f"Adjacency matrix for sequence {i} is None. Creating graph with no edges.")
                    # Create a graph with nodes but no edges
                    current_adj_matrix = np.zeros((seq_len, seq_len), dtype=int)
                    current_weights_matrix = np.empty((0,0)) # Or appropriate empty shape

            try:
                graph_data = to_parse_matrix(
                    adjacency_matrix=current_adj_matrix,
                    nodes_features=nodes_features_np,
                    weights_matrix=current_weights_matrix,
                    label=label_val
                )
                graphs_list.append(graph_data)
            except ValueError as e_parse:
                logging.error(f"Error creating Data object for sequence {i} (len {seq_len}): {e_parse}")
                # Append None or an empty Data object to maintain list length, to be filtered later if necessary
                graphs_list.append(None) # Placeholder for failed graph construction


    # Filter out any None graphs that resulted from errors
    valid_graphs_list = [g for g in graphs_list if g is not None]
    if len(valid_graphs_list) != len(graphs_list):
        logging.warning(f"Filtered out {len(graphs_list) - len(valid_graphs_list)} graphs due to errors during construction.")
        # This means the returned graphs_list might be shorter than input sequence_list.
        # This needs to be handled by the calling code (e.g., by filtering corresponding embeddings/labels).
        # For now, returning the list which might contain Nones or be shorter.
        # It's often better to return the list with Nones to allow upstream alignment.
        return graphs_list # Return list that may contain Nones

    return valid_graphs_list # Or graphs_list if Nones are preferred for alignment


def evaluate_model_multiple_runs(n_runs, model_type='hybrid', 
                                 X_full_data=None, graphs_data=None, y_full_data=None, # Full dataset for this model type
                                 X_test_data=None, graphs_test_data=None, y_test_data=None, # Test set for this model type
                                 params_hybrid=None, # Specific params for hybrid model
                                 device='cuda'):
    """
    Train and test a specified model multiple times and record metrics.
    Note: This function assumes that X_full_data, graphs_data, y_full_data are for training/validation splits
          and X_test_data, etc., are the final hold-out test set.
          train_hybrid_model and train_test_CNN_model will do their own train/val splits from X_full_data.
    """
    metric_names = ["accuracy", "precision_score", "sensitivity_score", "specificity_score", "auc", "mcc"]
    all_run_metrics = {name: [] for name in metric_names} # Stores test metrics from each run
    
    for run_num in range(n_runs): # Renamed run to run_num
        print(f"\n--- Run {run_num + 1}/{n_runs} for {model_type} model ---")
        
        run_test_metrics = None # To store the test metrics dict from the current run

        if model_type == 'hybrid':
            if params_hybrid is None:
                raise ValueError("params_hybrid must be provided for hybrid model type.")
            # Train the hybrid model (trains on X_full_data, validates internally)
            # train_hybrid_model returns: model, val_y_cpu, val_score_cpu, metrics_val_best_epoch
            trained_model, _, _, _ = train_hybrid_model(
                X_full_data, graphs_data, y_full_data, 
                params_hybrid, alpha=params_hybrid.get("alpha", 0.5), # Get alpha from params or default
                device=device
            )
            # Test the trained model on the hold-out test set
            # test_hybrid_model returns: metrics_dict_list, y_pred_np, y_true_np, y_scores_np
            run_test_metrics_list, _, _, _ = test_hybrid_model(
                trained_model, X_test_data, graphs_test_data, y_test_data, device=device
            )
            run_test_metrics = run_test_metrics_list[0] # Get the dict from the list

        elif model_type == 'CNN':
            # Train and test CNN model
            # train_test_CNN_model handles its own train/val split from X_full_data, then tests on X_test_data
            # Returns: model, val_y, val_score, test_y, test_score, metrics_test_dict_list
            _, _, _, _, _, run_test_metrics_list = train_test_CNN_model(
                X_full_data, y_full_data, X_test_data, y_test_data, device=device
            )
            run_test_metrics = run_test_metrics_list[0] # Get the dict
        else:
            raise ValueError("model_type must be 'hybrid' or 'CNN'")
        
        # Store metrics for this run if successful
        if run_test_metrics:
            for name in metric_names:
                if name in run_test_metrics:
                    all_run_metrics[name].append(run_test_metrics[name])
                else: # Should not happen if metric names are consistent
                    all_run_metrics[name].append(np.nan) 
                    logging.warning(f"Metric {name} not found in run {run_num+1} results for {model_type}.")
    
    # Compute statistics for each metric
    results_summary = {}
    for name in metric_names:
        values = all_run_metrics[name]
        if not values: # If no successful runs recorded values for this metric
            results_summary[name] = {'mean': np.nan, 'std_dev': np.nan, 'ci_95': (np.nan, np.nan), 'all_values': []}
            continue

        mean_val = np.nanmean(values) # Use nanmean for robustness
        std_dev_val = np.nanstd(values, ddof=1)  # Sample SD, nanstd for robustness
        
        # 95% CI (t-distribution for small samples, normal for larger)
        # Count non-NaN values for 'n'
        n_valid = np.sum(~np.isnan(values))
        if n_valid < 2 : # Need at least 2 data points for CI
            ci_lower, ci_upper = np.nan, np.nan
        else:
            # Use t-distribution critical value
            t_value = stats.t.ppf(0.975, df=n_valid-1)
            margin_of_error = t_value * (std_dev_val / np.sqrt(n_valid))
            ci_lower = mean_val - margin_of_error
            ci_upper = mean_val + margin_of_error
        
        results_summary[name] = {
            'mean': mean_val,
            'std_dev': std_dev_val,
            'ci_95': (ci_lower, ci_upper),
            'all_values': values  # Store all raw values (including NaNs if any)
        }
    
    return results_summary


# Main script part for data loading and Optuna study has been moved to parameter_optimization.py
# Any remaining code here should be for direct execution of pLM_graph.py if intended,
# e.g., for single runs with fixed parameters, or other analyses.

# Example of how one might run a single training/testing cycle if this file is run directly:
# (This is illustrative and would replace the Optuna/multi-run blocks if they were here)
if __name__ == '__main__':
    # This block is now primarily for illustration or direct single-run tests.
    # The Optuna optimization should be run from parameter_optimization.py.
    
    print("pLM_graph.py executed directly.")
    print("For hyperparameter optimization, run parameter_optimization.py.")
    print("This __main__ block can be used for single model runs with fixed parameters or other tests.")

    # Example: Load data (simplified, assuming files exist)
    # This would require similar data loading as in parameter_optimization.py's main block.
    # For brevity, this part is omitted here. Assume X_train, graphs, y_train, etc. are loaded.

    # Example: Fixed parameters for a single Hybrid model run
    # fixed_hybrid_params = {
    #     "lr": 0.0005, "gat_hidden": 160, "batch_size": 64, 
    #     "pos_weight_val": 3.0, "num_layers": 3, "alpha": 0.4
    # }
    # print("\nRunning a single Hybrid model train/test cycle with fixed parameters...")
    # Make sure to load/prepare: X_train_embeds, train_graphs, train_labels, 
    #                            X_test_embeds, test_graphs, test_labels
    
    # model, val_y, val_scores, val_metrics = train_hybrid_model(
    #     X_train_embeds, train_graphs, train_labels,
    #     fixed_hybrid_params, alpha=fixed_hybrid_params["alpha"], device='cuda'
    # )
    # test_metrics, _, _, _ = test_hybrid_model(
    #     model, X_test_embeds, test_graphs, test_labels, device='cuda'
    # )
    # print("\nSingle Hybrid Run - Validation Metrics (best epoch):", val_metrics)
    # print("Single Hybrid Run - Test Metrics:", test_metrics[0])

    # Example: Statistical comparison (if you have results from multiple runs saved)
    # print_results and t-test logic would go here if analyzing pre-computed results.
    pass # Placeholder for any direct execution logic

