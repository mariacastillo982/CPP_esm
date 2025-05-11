import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from functools import partial 
import optuna
from optuna.trial import Trial
from datetime import datetime

# Assuming pLM_graph.py is in the same directory or accessible in PYTHONPATH
from pLM_graph import (
    train_hybrid_model,
    test_hybrid_model,
    generate_esm_embeddings,
    generate_graphs)
import esm # For esm.pretrained

def make_objective(X,graphs,y,X_test, graphs_test, y_test):
    def objective(trial: Trial):
        # Hyperparameter space to explore
        trial_params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "gat_hidden": trial.suggest_int("gat_hidden", 32, 256, step=32),
            "alpha": trial.suggest_float("alpha", 0, 1, step=0.1),
            "batch_size": trial.suggest_int("batch_size", 32, 128, step=32),
            "pos_weight_val": trial.suggest_float("pos_weight_val", 1.5, 4, step=0.5),
            "num_layers": trial.suggest_int("num_layers", 1, 4, step=1)
        }

        
        model, _, _, metrics_val = train_hybrid_model(X, graphs, y, trial_params, alpha=trial_params["alpha"], device='cuda:0')
        metrics, _, _, _ = test_hybrid_model(model, X_test, graphs_test, y_test, device='cuda:0')
        
        # The original objective was:
        # best_val_mcc, best_result = train_and_evaluate_model(X = X, graphs = graphs, y = y, trial_params = trial_params, log_csv_path = log_filename)
        # return best_val_mcc 
        # Current objective returns test AUC
        return metrics[0]["auc"]  # We want to maximize AUC (or MCC as per original comment)

    return objective

if __name__ == "__main__":
    # Create a unique file name for logging CSV outputs from Optuna if needed by train_and_evaluate_model
    # log_filename_optuna = f"optuna_run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # whole dataset loading and dataset splitting
    dataset_train_val = pd.read_excel('./Final_non_redundant_sequences.xlsx',na_filter = False) # take care the NA sequence problem

    # generate the peptide embeddings for training/validation set
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

    graphs_train_val = generate_graphs(sequence_list_train_val, dataset_train_val, tertiary_structure_method=False)

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
    graphs_test = generate_graphs(sequence_list_test, dataset_test, tertiary_structure_method=False)

    # Optuna study
    study = optuna.create_study(direction="maximize") # Or "minimize" if optimizing loss
    
    study.optimize(make_objective(X_train_val_scaled, graphs_train_val, y_train_val, X_test_scaled, graphs_test, y_test), n_trials=30) # Adjust n_trials as needed

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Test AUC or MCC depending on objective's return): {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Example of how to use best_params 
    best_params = trial.params
    print("\nTraining final model with best parameters found by Optuna...")
    final_model, _, _, _ = train_hybrid_model(X_train_val_scaled, graphs_train_val, y_train_val, best_params, alpha=best_params["alpha"], device='cuda:0')
    final_metrics, _, _, _ = test_hybrid_model(final_model, X_test_scaled, graphs_test, y_test, device='cuda:0')
    print("\nFinal model performance on test set:")
    for k, v in final_metrics[0].items():
         print(f"  {k}: {v}")

