
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