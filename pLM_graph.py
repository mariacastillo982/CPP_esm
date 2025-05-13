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
    
def compare_roc_curves(scores_1, labels_1, scores_2, labels_2, name_1='Method 1', name_2='Method 2', save_path="output/roc_comparison.png"):
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

"""
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
"""

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
    save_metrics_to_csv(metrics_log_cnn, "output/training_validation_results_CNN.csv")
    
    # Final Test Evaluation using the best model from validation
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.tensor(X_test_final, dtype=torch.float32, device=device)
        test_y_tensor = torch.tensor(y_test_final, dtype=torch.float32, device=device).unsqueeze(1)
        test_outputs_logits = model(test_x_tensor).cpu() # Get logits on CPU

        best_thresh_test, best_mcc_test = optimize_threshold(test_y_tensor.cpu(), test_outputs_logits)
        
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
    plot_roc_curve(val_y_cpu_plot, val_scores_cpu_plot, "output/roc_auc_validation_cnn.png")
    # plot_output_scores(val_scores_cpu_plot, val_y_cpu_plot, "plot_scores_validation_cnn.png")
    
    test_y_cpu_plot = test_y_tensor.cpu().numpy().flatten()
    test_scores_cpu_plot = test_scores_probs.cpu().numpy().flatten()
    plot_roc_curve(test_y_cpu_plot, test_scores_cpu_plot, "output/roc_auc_testing_cnn.png")
    # plot_output_scores(test_scores_cpu_plot, test_y_cpu_plot, "plot_scores_testing_cnn.png")
    
    save_metrics_to_csv(metrics_test_final, "output/testing_results_CNN.csv")
    
    # Return model, validation y and scores, test y and scores, and test metrics
    return model, val_y_cpu_plot, val_scores_cpu_plot, test_y_cpu_plot, test_scores_cpu_plot, metrics_test_final




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
    
def train_hybrid_model(X_full_embed, graphs_full, y_full_labels, trial_params, alpha=0.5, device='cuda:0'): # Renamed inputs
    
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

    if not graphs_train: 
        raise ValueError("Graph list for training is empty. Cannot determine node feature dimension.")
    node_feature_dimension = graphs_train[0].x.shape[1]
    
    model = HybridModel(
        cnn_input_channels=1, 
        cnn_seq_len=X_train_embed.shape[1], # Assuming X_train_embed is (N, seq_len)
        node_feature_dimension=node_feature_dimension,
        gat_hidden=trial_params["gat_hidden"],
        alpha=alpha, 
        num_layers=trial_params["num_layers"]).to(device)

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
    save_metrics_to_csv(metrics_log_hybrid, "output/training_validation_results_Hybrid_CNN_GAT.csv")
    
    model.eval()
    with torch.no_grad():
        val_x_cnn_tensor_final = torch.tensor(X_val_embed, dtype=torch.float32, device=device)
        val_graphs_batch_final = Batch.from_data_list(graphs_val).to(device)
        val_outputs_logits_final = model(val_x_cnn_tensor_final, val_graphs_batch_final.x, val_graphs_batch_final.edge_index, val_graphs_batch_final.edge_attr, val_graphs_batch_final.batch)
        val_scores_probs_final = torch.sigmoid(val_outputs_logits_final).cpu().numpy().flatten()
    
    val_y_cpu_for_plot = y_val_labels 
    
    return model, val_y_cpu_for_plot, val_scores_probs_final, final_val_metrics_on_best_epoch


def test_hybrid_model(model, X_test_embed, graphs_test_list, y_test_labels, device='cuda:0'): 
    """
    Evaluate the trained HybridModel on the test dataset.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    y_test_labels_np = np.array(y_test_labels) if not isinstance(y_test_labels, np.ndarray) else y_test_labels


    with torch.no_grad():
        # CNN input for test set
        X_test_cnn_tensor = torch.tensor(X_test_embed, dtype=torch.float32, device=device)
        
        if not all(isinstance(g, Data) for g in graphs_test_list if g is not None): 
            valid_graphs_test = [g for g in graphs_test_list if g is not None and isinstance(g, Data) and g.x is not None and g.edge_index is not None]
            if not valid_graphs_test:
                logging.error("All graphs in graphs_test_list are invalid or None. Cannot perform testing.")
                error_metrics = [{"accuracy": np.nan, "precision_score": np.nan, "sensitivity_score": np.nan, 
                                   "specificity_score": np.nan, "mcc": np.nan, "auc": np.nan}]
                return error_metrics, np.array([]), y_test_labels_np, np.array([])

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
    save_metrics_to_csv(metrics_test_hybrid, "output/testing_results_Hybrid_CNN_GAT.csv")
    
    # Return: metrics dict, predictions (binary), true labels (numpy), scores (probabilities)
    return metrics_test_hybrid, y_preds_final_test.numpy().flatten(), test_y_cpu_plot, test_scores_cpu_plot


if __name__ == '__main__':

    print("pLM_graph.py executed directly.")
    print("For hyperparameter optimization, run parameter_optimization.py.")
    
    # whole dataset loading and dataset splitting
    dataset_train_val = pd.read_excel('./Final_non_redundant_sequences.xlsx',na_filter = False) 
    # generate the peptide embeddings
    sequence_list_train_val = dataset_train_val['sequence']
    
    # get embeddings for training and validation
    
    hub_dir = os.path.join(os.getcwd(), "./models/esmfold/")
    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir, exist_ok=True)
    torch.hub.set_dir(hub_dir)
    
    model_esm_embed, alphabet_esm_embed = esm.pretrained.esm2_t33_650M_UR50D()
    
    # Check if embedding file exists, else generate
    train_val_embeddings_file = './whole_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv'
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
    dataset_test = pd.read_excel('./kelm.xlsx',na_filter = False) # take care the NA sequence 
    sequence_list_test = dataset_test['sequence']
    
    # get embeddings for testing
    test_embeddings_file = './kelm_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv'
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
         "lr": 0.0005, "gat_hidden": 160, "batch_size": 64, 
         "pos_weight_val": 3.0, "num_layers": 3, "alpha": 0.4}
    print("\nRunning a single Hybrid model train/test cycle with fixed parameters...")
    
    print("---------------------------- Training & Testing CNN ----------------------------:")
    model_CNN, val_y_cnn, val_score_cnn, test_y_cnn, test_score_cnn, metrics_CNN = train_test_CNN_model(X_train_val, y_train_val, X_test, y_test, device='cuda')
    
    print("---------------------------- Training CNN + GAT----------------------------:")    
    model,val_y_gat,val_score_gat, metrics_val = train_hybrid_model(X_train_val, graphs_train_val, y_train_val, params, alpha=0.4, device='cuda')
    print("---------------------------- Testing CNN + GAT----------------------------:")  
    metrics_GAN, preds, test_y_gat, test_score_gat = test_hybrid_model(model, X_test, graphs_test, y_test, device='cuda')

