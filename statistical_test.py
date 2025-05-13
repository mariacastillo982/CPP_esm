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
#from torch.utils.tensorboard import SummaryWriter
from itertools import product
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import ttest_rel
from graph.construct_graphs import generate_graphs
from models.esm2.esm2_model_handler import generate_esm_embeddings
from models.GAT.GAT import Conv1DClassifier, GATModel
from graph.tertiary_structure_handler import load_tertiary_structures, predict_tertiary_structures
from pLM_graph import train_hybrid_model, test_hybrid_model, train_test_CNN_model

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
        
        

        
if __name__ == '__main__':

    print("Statistical testing using the non-parametric bootstrap")    
    # whole dataset loading and dataset splitting
    dataset_train_val = pd.read_excel('input/Final_non_redundant_sequences.xlsx',na_filter = False) 
    # generate the peptide embeddings
    sequence_list_train_val = dataset_train_val['sequence']
    
    # get embeddings for training and validation
    
    hub_dir = os.path.join(os.getcwd(), "./models/esmfold/")
    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)
    
    model_esm_embed, alphabet_esm_embed = esm.pretrained.esm2_t33_650M_UR50D()
    
    # Check if embedding file exists, else generate
    train_val_embeddings_file = 'input/whole_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv'
    if os.path.exists(train_val_embeddings_file):
        print(f"Loading existing train/val embeddings from {train_val_embeddings_file}")
        X_train_val_data = pd.read_csv(train_val_embeddings_file,header=0, index_col = 0,delimiter=',')
    else:
        print(f"Generating train/val embeddings and saving to {train_val_embeddings_file}")
        X_train_val_data = generate_esm_embeddings(model_esm_embed, alphabet_esm_embed, sequence_list_train_val, train_val_embeddings_file)
    
    X_train_val = np.array(X_train_val_data)
    y_train_val = np.array(dataset_train_val['label'])
    
    graphs_train_val = generate_graphs(sequence_list_train_val, dataset_train_val, tertiary_structure_method=False, pdb_path = Path('./output/ESMFold_pdbs/'))
    
    # Load test dataset
    dataset_test = pd.read_excel('input/kelm.xlsx',na_filter = False) # take care the NA sequence 
    sequence_list_test = dataset_test['sequence']
    
    # get embeddings for testing
    test_embeddings_file = 'input/kelm_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv'
    if os.path.exists(test_embeddings_file):
        print(f"Loading existing test embeddings from {test_embeddings_file}")
        X_test_data = pd.read_csv(test_embeddings_file,header=0, index_col = 0,delimiter=',')
    else:
        print(f"Generating test embeddings and saving to {test_embeddings_file}")
        X_test_data = generate_esm_embeddings(model_esm_embed, alphabet_esm_embed, sequence_list_test, test_embeddings_file)
    
    X_test = np.array(X_test_data)
    y_test = np.array(dataset_test['label'])

    # Normalize the data 
    scaler = MinMaxScaler()
    scaler.fit(X_train_val) # Fit scaler ONLY on the training/validation portion that Optuna will see and split.
    
    X_train_val_scaled = scaler.transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)

    # get graphs for testing
    graphs_test = generate_graphs(sequence_list_test, dataset_test, tertiary_structure_method=False, pdb_path = Path('./output/ESMFold_pdbs_kelm/'))
    
    params = {
         "lr": 0.000572, "gat_hidden": 160, "batch_size": 96, 
         "pos_weight_val": 3.5, "num_layers": 3, "alpha": 0.4}
    n_runs = 30  # Number of runs (recommended: 10-30)

    print("\n==== Evaluating Hybrid Model (CNN + GAT) ====")
    hybrid_results = evaluate_model_multiple_runs(
        n_runs=n_runs,
        model_type='hybrid',
        X_train=X_train_val_scaled,
        graphs=graphs_train_val,
        y_train=y_train_val,
        X_test=X_test_scaled,
        graphs_test=graphs_test,
        y_test=y_test,
        params=params,
        device='cuda'
    )

    print("\n==== Evaluating CNN Model ====")
    cnn_results = evaluate_model_multiple_runs(
        n_runs=n_runs,
        model_type='CNN',
        X_train=X_train_val_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
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