# Protein Language Model Graph (pLM-Graph) Project

This repository contains the code for a project that utilizes protein language models (ESM2) and graph neural networks (GNNs) for protein-related prediction tasks, likely focusing on binary classification (e.g., identifying certain protein properties). It includes functionalities for ESM2 embedding generation, graph construction, model training (CNN-only and a hybrid CNN+GAT model), hyperparameter optimization, and statistical evaluation of model performance.

## Table of Contents

1.  [Installation](#installation)
2.  [Data Preparation](#data-preparation)
3.  [Usage and Reproducing Results](#usage-and-reproducing-results)
    *   [3.1. Hyperparameter Optimization](#31-hyperparameter-optimization)
    *   [3.2. Training and Testing a Single Model Instance](#32-training-and-testing-a-single-model-instance)
    *   [3.3. Statistical Significance Testing](#33-statistical-significance-testing)
4.  [Output Files](#output-files)
5.  [Directory Structure](#directory-structure)
6.  [Dependencies](#dependencies)

## 1. Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Set up Conda Environment:**
    Use the `environment.yml` file to create a Conda environment with all necessary dependencies.
    ```bash
    conda env create -f environment.yml
       conda activate CPP_pred # Name specified in environment.yml
    ```

3.  **CUDA (for GPU usage):**
    The scripts are configured to use CUDA if available (e.g., `device='cuda:0'`). Ensure you have a compatible NVIDIA driver and CUDA toolkit installed if you plan to use a GPU. The PyTorch version in `environment.yml` is CUDA-enabled.

4.  **Key libraries include:**

*   Python 3.7
*   PyTorch 
*   PyTorch Geometric 
*   ESM (Facebook Research) (`fair-esm`)
*   Optuna
*   Pandas
*   NumPy
*   Scikit-learn 
*   Matplotlib
*   SciPy
*   BioPython
*   tqdm
*   tensorboardX (or `torch.utils.tensorboard`)

All dependencies are listed in `environment.yml`. 

## 2. Data Preparation

1.  **Input Datasets:**
    *   Create an `input/` directory in the root of the repository if it doesn't exist.
    *   Place your training/validation dataset (e.g., `Final_non_redundant_sequences.xlsx`) in the `input/` directory.
    *   Place your test dataset (e.g., `kelm.xlsx`) in the `input/` directory.
    *   These files should contain columns named 'sequence' (for protein sequences) and 'label' (for binary labels).

2.  **ESM2 Embeddings:**
    *   All scripts (`pLM_graph.py`, `parameter_optimization.py`, `statistical_test.py`) will automatically generate ESM2 embeddings (using `esm2_t33_650M_UR50D`) for your sequences if cached embedding files are not found in the `input/` directory.
    *   By default, these scripts look for/save embeddings to:
        *   `input/whole_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv` (for training/validation from `Final_non_redundant_sequences.xlsx`)
        *   `input/kelm_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv` (for testing from `kelm.xlsx`)
    *   The ESM2 model for embeddings will be downloaded via `torch.hub`. The scripts configure `torch.hub.set_dir(os.path.join(os.getcwd(), "./models/esmfold/"))`, so models will be cached there (or `~/.cache/torch/hub` if that path isn't writable/preferred by hub).

3.  **Graph Data:**
    *   Graphs are generated on-the-fly by `graph.construct_graphs.generate_graphs`.
    *   The scripts currently use `tertiary_structure_method=False`, meaning graph edges are likely derived from sequence information or other non-PDB sources defined in `graph.edge_construction_functions.py`.
    *   If `tertiary_structure_method=True` were used, PDB files would be predicted by ESMFold (via `graph.tertiary_structure_handler.py`).
        *   `pLM_graph.py` and `statistical_test.py` specify saving these to `output/ESMFold_pdbs/` (for train/val) and `output/ESMFold_pdbs_kelm/` (for test).
        *   `parameter_optimization.py` currently uses the default `pdb_path` from `construct_graphs.py` (which is `output/ESMFold_pdbs/`) for both train/val and test sets when `tertiary_structure_method=True`.
    *   The ESMFold model would also be downloaded via `torch.hub`.

## 3. Usage and Reproducing Results

Ensure your Conda environment (`esm-axp-gdl-env`) is activated. All commands should generally be run from the root directory of the repository. Make sure the `input/` directory is populated as described above and an `output/` directory exists or can be created by the scripts.

### 3.1. Hyperparameter Optimization

To find optimal hyperparameters for the Hybrid (CNN+GAT) model for retraining or adapting to new data:

1.  **Purpose:** Uses Optuna to search for optimal hyperparameters (learning rate, GAT hidden dimensions, `alpha` for combining CNN and GAT outputs, batch size, positive class weight for BCE loss, number of GAT layers). The objective is to maximize a target metric (currently test AUC) over a specified number of trials.
2.  **Command:**
    ```bash
    python parameter_optimization.py
    ```
3.  **Process:**
    *   Loads data from the `input/` directory, generates/loads embeddings from/to `input/`, and constructs graphs.
    *   Splits the "train_val" data internally for training and validation within each Optuna trial.
    *   Evaluates performance on the "test" set to guide optimization.
4.  **Output:**
    *   Prints Optuna study progress.
    *   Displays the best trial's value (metric) and corresponding hyperparameters.
    *   Trains and tests a final model using these best parameters and prints its performance.

### 3.2. Training and Testing a Single Model Instance

To train and evaluate the CNN-only model and the Hybrid (CNN+GAT) model using a fixed set of hyperparameters (as defined in the `if __name__ == '__main__':` block of `pLM_graph.py`):

1.  **Purpose:** Perform a single training and testing cycle for both the standalone CNN model and the Hybrid CNN+GAT model. This is useful for quick evaluations or reproducing results with known parameters.
2.  **Command:**
    ```bash
    python pLM_graph.py
    ```
3.  **Process:**
    *   Loads training/validation data (`input/Final_non_redundant_sequences.xlsx`) and test data (`input/kelm.xlsx`).
    *   Generates/loads ESM2 embeddings from/to `input/` and graph structures. Normalizes features.
    *   **CNN Model:** Trains `Conv1DClassifier`. Outputs training progress, validation metrics (including MCC optimization for threshold), and final test metrics. Saves results and ROC curves to the `output/` directory.
    *   **Hybrid Model (CNN+GAT):** Trains `HybridModel` using predefined hyperparameters (e.g., `lr: 0.000572, gat_hidden: 160, alpha: 0.4`, etc., taken from `params` dictionary). Outputs training progress, validation metrics, and final test metrics. Saves results to the `output/` directory. ROC curve plotting for the hybrid model on the test set is available in `test_hybrid_model` but might be commented out by default.
4.  **Output:**
    *   Console logs: Training progress and evaluation metrics.
    *   CSV files in `output/`:
        *   `output/training_validation_results_CNN.csv`
        *   `output/testing_results_CNN.csv`
        *   `output/training_validation_results_Hybrid_CNN_GAT.csv`
        *   `output/testing_results_Hybrid_CNN_GAT.csv`
    *   ROC curve plots in `output/`:
        *   `output/roc_auc_validation_cnn.png`
        *   `output/roc_auc_testing_cnn.png`
        *   (Hybrid model ROC plots can be enabled if needed, e.g., `output/roc_auc_testing_hybrid.png`).

### 3.3. Statistical Significance Testing

To reproduce statistical test results comparing the Hybrid model against the CNN-only baseline over multiple runs:

1.  **Purpose:** Assesses the statistical significance of performance differences between the Hybrid model and the CNN model. It trains and tests each model `n_runs` times (default 30), collects metrics, calculates mean, standard deviation, and 95% confidence intervals, and performs a paired t-test.
2.  **Command:**
    ```bash
    python statistical_test.py
    ```
3.  **Process:**
    *   Loads data and prepares embeddings/graphs similarly to `pLM_graph.py` (using the `input/` directory).
    *   Uses the same fixed hyperparameters for the Hybrid model as in `pLM_graph.py`.
    *   For `n_runs`:
        *   Trains and tests the Hybrid model, collecting its test metrics.
        *   Trains and tests the CNN model, collecting its test metrics.
        *   Each run involves a fresh train/validation split from the full training data, and evaluation on the fixed test set.
    *   Calculates and prints aggregate statistics (mean, SD, 95% CI) for each metric for both models.
    *   Performs a paired t-test on the collected metrics for each run to compare the two models.
4.  **Output:**
    *   Console logs for each run.
    *   Summary statistics (Mean, SD, 95% CI) for both models for metrics: accuracy, precision, sensitivity, specificity, AUC, MCC.
    *   P-values from paired t-tests for each metric, indicating if performance differences are statistically significant.

## 4. Output Files

Most output files are saved in the `output/` directory. This directory will be created by the scripts if it doesn't exist.

*   **Metrics (CSV):**
    *   `output/training_validation_results_CNN.csv`
    *   `output/testing_results_CNN.csv`
    *   `output/training_validation_results_Hybrid_CNN_GAT.csv`
    *   `output/testing_results_Hybrid_CNN_GAT.csv`
    *(Note: `statistical_test.py` also calls training/testing functions which might overwrite these files from the last run of its multiple runs. For per-run metrics from `statistical_test.py`, you might need to modify the script to save them uniquely or aggregate differently if needed beyond the printed summary.)*
*   **Plots (PNG):**
    *   `output/roc_auc_validation_cnn.png`
    *   `output/roc_auc_testing_cnn.png`
    *   `output/roc_auc_testing_hybrid.png` (if enabled in `test_hybrid_model` within `pLM_graph.py`)
*   **Cached Embeddings (CSV, in `input/`):**
    *   `input/whole_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv`
    *   `input/kelm_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv`
*   **Predicted PDBs (if `tertiary_structure_method=True` was used):**
    *   `output/ESMFold_pdbs/`
    *   `output/ESMFold_pdbs_kelm/`


