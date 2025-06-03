import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add project root to Python path to enable 'main' module imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from surrogate_dependent_gpr import DependentGPRModels, MultitaskGPModel, ExactGPModel, SparseGPModel
from surrogate_independent_gpr import IndependentGPRModels
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time




def plot_predictions_with_uncertainty(model, x_true, y_true, y_predicted, var_predicted, path):
    
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    fig = plt.figure()
    # fig.suptitle(f"Predictions with 95% confidence interval using err", fontsize=14)
    ax1 = fig.add_subplot(211)
    ax1.set_title("Predictions with 95% confidence interval using error")
    ax2 = fig.add_subplot(212)
    ax2.set_title("Errors")
    mae_list = []
    rmse_list = []
    r2_list = []
    for i_dim in range(y_train.shape[1]):
        # Make sure all arrays are 1D for plotting
        # y_train_flat = y_train[:, i_dim].numpy().flatten() if torch.is_tensor(y_train) else y_train[:, i_dim].flatten()
        # y_hat_train_flat = y_hat_train[:, i_dim].flatten()            
        y_test_flat = y_true[:, i_dim].numpy().flatten() if torch.is_tensor(y_true) else y_true[:, i_dim].flatten()
        y_hat_test_flat = y_predicted[:, i_dim].flatten()

        # 95% confidence interval
        y_error = 2*np.sqrt(var_predicted[:, i_dim].flatten())
        ax1.errorbar(y_test_flat, y_hat_test_flat, yerr=y_error, fmt='x', c='r', 
                    ecolor='r', alpha=0.3)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_test_flat - y_hat_test_flat))
        mae_list.append(mae)
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((y_test_flat - y_hat_test_flat)**2))
        rmse_list.append(rmse)
        # R2 Score
        r2 = r2_score(y_test_flat, y_hat_test_flat)
        r2_list.append(r2)
        ax2.plot([0, 1, 2],[mae, rmse, r2], '-o', c='r', label=f"Wall Depth {i_dim}")

    mae_mean = np.mean(mae_list)
    rmse_mean = np.mean(rmse_list)
    r2_mean = np.mean(r2_list)
    
    # Create comprehensive visualizations
    n_depths = len(mae_list)
    
    # 1. Bar plots for each metric
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('Performance Metrics by Wall Depth', fontsize=16)
    
    # Plot MAE for each wall depth
    depths = [f'Depth {i}' for i in range(n_depths)]
    ax1.bar(depths, mae_list, color='skyblue')
    ax1.axhline(mae_mean, color='r', linestyle='--', label=f'Mean: {mae_mean:.4f}')
    ax1.set_xlabel('Wall Depth Index')
    ax1.set_ylabel('MAE')
    ax1.legend()
    ax1.set_xticklabels(depths, rotation=45)
    
    # Plot RMSE for each wall depth
    ax2.bar(depths, rmse_list, color='lightgreen')
    ax2.axhline(rmse_mean, color='r', linestyle='--', label=f'Mean: {rmse_mean:.4f}')
    ax2.set_xlabel('Wall Depth Index')
    ax2.set_ylabel('RMSE')
    ax2.legend()
    ax2.set_xticklabels(depths, rotation=45)
    
    # Plot R2 for each wall depth
    ax3.bar(depths, r2_list, color='salmon')
    ax3.axhline(r2_mean, color='r', linestyle='--', label=f'Mean: {r2_mean:.4f}')
    ax3.set_xlabel('Wall Depth Index')
    ax3.set_ylabel('R² Score')
    ax3.set_ylim(0, 1)  # R² typically ranges from 0 to 1
    ax3.legend()
    ax3.set_xticklabels(depths, rotation=45)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save bar plots
    pp = PdfPages(path.with_stem(path.stem + "_metrics_bar"))
    pp.savefig(fig1)
    pp.close()
    
    png_path = path.with_stem(path.stem + "_metrics_bar").with_suffix('.png')
    plt.savefig(png_path)
    plt.close(fig1)
    
    # 2. Radar chart - REMOVED
    
    # 3. Boxplots for metrics distribution
    fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Distribution of Performance Metrics Across Wall Depths', fontsize=16)
    
    ax1.boxplot(mae_list)
    ax1.set_ylabel('MAE')
    ax1.set_xticks([1])
    ax1.set_xticklabels([''])
    ax1.set_title(f'Mean: {mae_mean:.4f}')
    
    ax2.boxplot(rmse_list)
    ax2.set_ylabel('RMSE')
    ax2.set_xticks([1])
    ax2.set_xticklabels([''])
    ax2.set_title(f'Mean: {rmse_mean:.4f}')
    
    ax3.boxplot(r2_list)
    ax3.set_ylabel('R² Score')
    ax3.set_xticks([1])
    ax3.set_xticklabels([''])
    ax3.set_title(f'Mean: {r2_mean:.4f}')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save boxplots
    pp = PdfPages(path.with_stem(path.stem + "_metrics_box"))
    pp.savefig(fig3)
    pp.close()
    
    png_path = path.with_stem(path.stem + "_metrics_box").with_suffix('.png')
    plt.savefig(png_path)
    plt.close(fig3)
    
    # 4. Scatter plots for predicted vs actual with R² values
    n_cols = min(3, n_depths)
    n_rows = (n_depths + n_cols - 1) // n_cols  # Ceiling division to get number of rows needed
    
    fig4, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    fig4.suptitle('Predicted vs Actual Values by Wall Depth', fontsize=16)
    
    # Handle axes properly whether it's a single subplot or array of subplots
    if n_depths == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(n_depths):
        ax = axes[i]
        ax.scatter(y_test[:, i], y_predicted[:, i], alpha=0.5)
        ax.axline((0, 0), slope=1, c='r', linestyle='--')
        
        # Calculate min and max for equal axis limits
        min_val = min(y_test[:, i].min(), y_predicted[:, i].min())
        max_val = max(y_test[:, i].max(), y_predicted[:, i].max())
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Wall Depth {i}\nR²: {r2_list[i]:.3f}')
    
    # Hide any unused subplots
    for i in range(n_depths, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save scatter plots
    pp = PdfPages(path.with_stem(path.stem + "_scatter"))
    pp.savefig(fig4)
    pp.close()
    
    png_path = path.with_stem(path.stem + "_scatter").with_suffix('.png')
    plt.savefig(png_path)
    plt.close(fig4)

def plot_wall(model, x_true, y_true, y_predicted, var_predicted, path, depth_lims=[-10, 0]):
    
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    # Set up figure
    # fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig = plt.figure(figsize=(10,10))
    nr_depths = y_true.shape[1]
    # Use a single depth scale for both plots
    depths = np.linspace(depth_lims[0], depth_lims[1], nr_depths)  # Simplified depth representation
        
    # Convert tensors to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.numpy()

    if torch.is_tensor(y_predicted):
        y_predicted = y_predicted.numpy()

    if torch.is_tensor(var_predicted):
        var_predicted = var_predicted.numpy()

    added_value = 1000
    # Calculate residuals
    residuals = (y_true - y_predicted) - added_value

    # Get statistics
    mean_error = np.mean(residuals, axis=0)
    std_error = np.std(residuals, axis=0)
    
    # Only plot a sample of individual residuals to reduce clutter
    # max_lines = min(20, len(residuals))
    # indices = np.linspace(0, len(residuals) - 1, max_lines, dtype=int)
    
    # for i in indices:
        # ax.plot(residuals[i], x_true[:, -1], c='b', alpha=0.1)
    
    ax = fig.add_subplot(221)
    # Plot mean with confidence band
    ax.plot(mean_error, depths, c='r', linewidth=2, label='Mean Error')
    ax.fill_betweenx(depths, 
                    mean_error - 2*std_error, 
                    mean_error + 2*std_error, 
                    color='r', alpha=0.2, label='95% confidence interval')
        
    # Labels and styling
    ax.set_xlabel('Error [mm]', fontsize=12)
    ax.set_ylabel('Depth [m]', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()

    ax2 = fig.add_subplot(222)
    # for i_depth in range(nr_depths):
    n_samples = min(100, y_true.shape[0])
    # for i_sample in range(n_samples):
        # ax2.plot(y_true[i_sample,:], depths,  c='g', linewidth=1, alpha=0.1)

    y_true_mean = y_true.mean(axis=0) - added_value
    y_true_std = y_true.std(axis=0)

    y_pred_mean = y_predicted.mean(axis=0) - added_value
    # y_pred_std = y_predicted.std(axis=0)
    y_error = 2*np.sqrt(var_predicted)
    y_pred_std = y_error.mean(axis=0)

    # plot original wall position
    ax2.vlines(0, depths[0], depths[-1], color='k', linestyle='-', linewidth=1, label='Original wall position')

    ax2.plot(y_true_mean, depths, c='g', linewidth=2, label='Mean Dsheet Displacement')
    ax2.plot(y_pred_mean, depths, '--', c='y', linewidth=2, label='Mean GPR Predicted Displacement')

    # plot upper and lower true bounds as lines
    # ax2.plot(y_true_mean + 2*y_true_std, depths, '--', c='g', linewidth=1, label='True bounds')
    # ax2.plot(y_true_mean - 2*y_true_std, depths, '--', c='g', linewidth=1)

    ax2.fill_betweenx(depths, y_pred_mean - 2*y_pred_std, y_pred_mean + 2*y_pred_std, color='y', alpha=0.2, label='95% confidence interval')
    ax2.set_xlabel('Displacement [mm]', fontsize=12)
    ax2.set_ylabel('Depth [m]', fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    for sample in range(n_samples):
        ax3.plot(y_true[sample,:], depths, c='g', linewidth=1, alpha=0.1)
        ax4.plot(y_predicted[sample,:], depths, c='y', linewidth=1, alpha=0.1)
    ax3.set_xlabel('True Displacement [mm]', fontsize=12)
    ax3.set_ylabel('Depth [m]', fontsize=12)
    ax3.grid(alpha=0.3)
    ax3.legend()

    ax4.set_xlabel('Predicted Displacement [mm]', fontsize=12)
    ax4.set_ylabel('Depth [m]', fontsize=12)
    ax4.grid(alpha=0.3) 
    ax4.legend()

    # Adjust layout
    plt.tight_layout()
    
    # Save as PDF and PNG
    pp = PdfPages(path)
    pp.savefig(fig)
    pp.close()
    
    png_path = path.with_suffix('.png')
    plt.savefig(png_path)
    plt.close(fig)


def plot_variables(model, x_true, y_true, y_predicted, var_predicted, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
    residuals = y_true - y_predicted
    st_residuals = residuals / np.abs(y_true+1e-8) * 100
    
    figs = []
    nr_parameters = x_true.shape[1]
    # create grid based on nr_parameters
    nr_rows = int(np.ceil(np.sqrt(nr_parameters)))
    nr_cols = int(np.ceil(nr_parameters / nr_rows))
    fig = plt.figure(figsize=(nr_cols*5, nr_rows*5))
    for parameter in range(nr_parameters):
        ax = fig.add_subplot(nr_rows, nr_cols, parameter+1)
        # fig.suptitle(f"Parameter #{parameter+1:d}", fontsize=14)
        ax.set_title(f"Parameter #{parameter+1:d}", fontsize=14)

        for wall_depth in range(y_true.shape[1]):
            sc = ax.scatter(x_true[:, parameter], st_residuals[:, wall_depth], c=x_true[:, parameter])
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Parameter}", fontsize=10, labelpad=10)
        ax.set_xlabel('Parameter', fontsize=12)
        ax.set_ylabel('Error [%]', fontsize=12)
        ax.set_ylim(residuals.min(), residuals.max())
        ax.grid()
    figs.append(fig)
        
    pp = PdfPages(path)
    [pp.savefig(fig) for fig in figs]
    pp.close()

    # Save as PNG
    png_path = path.with_suffix('.png')
    plt.savefig(png_path)
    plt.close(fig)

def plot_training_and_test_data(X_train, X_test, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
    nr_parameters = X_train.shape[1]
    fig = plt.figure(figsize=(15, nr_parameters* 5))
    fig.suptitle('Distribution of Parameters in Training and Test Sets', fontsize=16)
    
    parameter_names = ['Klei_soilphi', 'Klei_soilcohesion', 'Klei_soilcurkb1',
                        'Zand_soilphi', 'Zand_soilcurkb1',
                        'Zandlos_soilphi', 'Zandlos_soilcurkb1',
                        'Zandvast_soilphi', 'Zandvast_soilcurkb1',
                        'Wall_SheetPilingElementEI']
    
    for parameter in range(nr_parameters):
        param_name = parameter_names[parameter] if parameter < len(parameter_names) else f'Parameter {parameter+1}'
        
        # Create shared bins for consistent comparison
        min_val = min(X_train[:, parameter].min(), X_test[:, parameter].min())
        max_val = max(X_train[:, parameter].max(), X_test[:, parameter].max())
        bins = np.linspace(min_val, max_val, 25)
        
        # Training data histogram
        ax1 = fig.add_subplot(nr_parameters, 2, 2*parameter+1)
        ax1.hist(X_train[:, parameter], bins=bins, color='skyblue', edgecolor='white', 
                alpha=0.7, label='Training Data')
        ax1.set_title(f'{param_name} - Training Data', fontsize=12)
        ax1.set_xlabel('Value', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.legend()

        # Test data histogram
        ax2 = fig.add_subplot(nr_parameters, 2, 2*parameter+2)
        ax2.hist(X_test[:, parameter], bins=bins, color='salmon', edgecolor='white', 
                alpha=0.7, label='Test Data')
        ax2.set_title(f'{param_name} - Test Data', fontsize=12)
        ax2.set_xlabel('Value', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.legend()

        # Set y-axis limits to be the same for proper comparison
        y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(0, y_max)
        ax2.set_ylim(0, y_max)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for the suptitle
    fig.savefig(path)
    png_path = path.with_suffix('.png')
    plt.savefig(png_path)
    plt.close(fig)


def plot(model, x_test, y_test, y_pred, y_var, X_train, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
    # y_hat_train, var_train = model.predict(x_train)
    # y_hat_test, var_test = model.predict(x_test)
    # y_hat_test = np.squeeze(y_hat_test)
    # var_test = np.squeeze(var_test)
    
    plot_predictions_with_uncertainty(model, x_test, y_test, y_pred, y_var, path/"predictions.pdf")
    plot_wall(model, x_test, y_test, y_pred, y_var, path/"wall.pdf")
    # plot_variables(model, x_test, y_test, y_pred, y_var, path/"variables.pdf")
    plot_training_and_test_data(X_train, x_test, path/"training_and_test_data.pdf")


def compute_metrics(y_true, y_pred, prediction_time):
    """
    Compute performance metrics for model evaluation.
    """
    # Mean Squared Error per output dimension
    mse_per_dim = np.mean((y_true - y_pred)**2, axis=0)
    
    # Root Mean Squared Error per output dimension
    rmse_per_dim = np.sqrt(mse_per_dim)
    
    # Mean Absolute Error per output dimension
    mae_per_dim = np.mean(np.abs(y_true - y_pred), axis=0)
    
    # R² score per output dimension
    r2_per_dim = np.array([r2_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])

    # Standardized mean absolute error per output dimension
    nrmse_per_dim = mae_per_dim / (y_true.max(axis=0) - y_true.min(axis=0))
    
    # Overall metrics
    overall_mse = np.mean(mse_per_dim)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = np.mean(mae_per_dim)
    overall_r2 = np.mean(r2_per_dim)
    overall_nrmse = np.mean(nrmse_per_dim)
    
    return {
        'mse_per_dim': list(mse_per_dim),
        'rmse_per_dim': list(rmse_per_dim),
        'mae_per_dim': list(mae_per_dim),
        'r2_per_dim': list(r2_per_dim),
        'nrmse_per_dim': list(nrmse_per_dim),
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2,
        'overall_nrmse': overall_nrmse,
        'prediction_time': prediction_time
    }

def save_predictions(model, X_test_tensor, plot_dir, model_path):
    # predict on test data
    # start timer
    start_time = time.time()
    y_hat_test = []
    var_test = []
    # for i in range(X_test_tensor.shape[0]):
    #     model = DependentGPRModels()
    #     model.load(model_path)
    #     print(f"Shape of X_test_tensor[i]: {X_test_tensor[i].unsqueeze(0).shape}")
    #     y_hat_test_i, var_test_i = model.predict(X_test_tensor[i].unsqueeze(0))
    #     y_hat_test.append(y_hat_test_i)
    #     var_test.append(var_test_i)
    # print(f"y_hat_test.shape: {y_hat_test.shape}")
    # print(f"var_test.shape: {var_test.shape}")
    # print(50*'-=-')

    model = DependentGPRModels()
    model.load(model_path)
    y_hat_test, var_test = model.predict(X_test_tensor)
    y_hat_test = np.array(y_hat_test)
    var_test = np.array(var_test)
    end_time = time.time()


    prediction_time = end_time - start_time

    y_hat_test = np.squeeze(y_hat_test)
    var_test = np.squeeze(var_test)
    predictions = {}
    var_predictions = {}
    for i in range(y_hat_test.shape[1]):
        # df = pd.DataFrame({"y_hat_test": y_hat_test[:, i], "var_test": var_test[:, i]})
        predictions[f"y_hat_test_{i}"] = y_hat_test[:, i]
        var_predictions[f"var_test_{i}"] = var_test[:, i]
    df = pd.DataFrame(predictions)
    df.to_csv(plot_dir/"predictions.csv", index=False)
    df = pd.DataFrame(var_predictions)
    df.to_csv(plot_dir/"var_predictions.csv", index=False)


    metrics = compute_metrics(y_test, y_hat_test, prediction_time)
    # save metrics to file
    with open(plot_dir/"result_metrics.json", "w") as f:
        json.dump(metrics, f)

def load_predictions(plot_dir):
    df = pd.read_csv(plot_dir/"predictions.csv")
    df_var = pd.read_csv(plot_dir/"var_predictions.csv")
    return df, df_var

def plot_comparison_multitask_gpr_models(result_metrics_paths, plot_dir):
    # load all result_metrics.json files
    result_metrics = []
    model_names = []
    ranks = []
    test_data_sizes = []
    train_data_sizes = []
    
    for path in result_metrics_paths:
        with open(path, "r") as f:
            metrics = json.load(f)
            result_metrics.append(metrics)
            
        # Extract model information from path
        path_str = str(path)
        parts = path_str.split('/')[-2].split('_')
        
        # Extract rank (last part)
        rank = parts[-1]
        ranks.append(int(rank))
        
        # Extract test data size (after Test_Data_)
        test_data_idx = parts.index('Data') + 1 if 'Data' in parts else -1
        test_data_size = parts[test_data_idx] if test_data_idx > 0 and test_data_idx < len(parts) else "Unknown"
        test_data_sizes.append(int(test_data_size))
        
        # Extract training data size (second-to-last part)
        train_data_size = parts[-2]
        train_data_sizes.append(int(train_data_size))
        
        model_names.append(f"R{rank}, Train:{train_data_size}, Test:{test_data_size}")
    
    if not isinstance(plot_dir, Path):
        plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract metrics for comparison
    maes = [m['overall_mae'] for m in result_metrics]
    rmses = [m['overall_rmse'] for m in result_metrics]
    r2s = [m['overall_r2'] for m in result_metrics]
    prediction_times = [m['prediction_time'] for m in result_metrics]
    
    # Create unique lists for grouping
    unique_ranks = sorted(list(set(ranks)))
    unique_test_sizes = sorted(list(set(test_data_sizes)))
    unique_train_sizes = sorted(list(set(train_data_sizes)))
    
    # Create a color scale for train sizes and marker styles for test sizes
    train_colors = {size: plt.cm.tab10(i/len(unique_train_sizes)) 
                   for i, size in enumerate(unique_train_sizes)}
    
    test_markers = {size: marker for size, marker in 
                   zip(unique_test_sizes, ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*'])}
    
    # Create a 3D grid for rank, train size, test size
    # -------------------------------------------------------------------------
    # 1. Heatmap of metrics by train and test sizes (for each rank)
    for metric_name, metric_values in [
        ('MAE', maes),
        ('RMSE', rmses),
        ('R²', r2s)
    ]:
        for rank in unique_ranks:
            # Create a matrix of values for this rank
            train_test_matrix = np.zeros((len(unique_train_sizes), len(unique_test_sizes)))
            train_test_values = {}
            
            for i, (r, train_size, test_size, value) in enumerate(zip(ranks, train_data_sizes, test_data_sizes, metric_values)):
                if r == rank:
                    train_idx = unique_train_sizes.index(train_size)
                    test_idx = unique_test_sizes.index(test_size)
                    train_test_matrix[train_idx, test_idx] = value
                    train_test_values[(train_size, test_size)] = value
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(train_test_matrix, cmap='viridis_r' if metric_name == 'R²' else 'viridis')
            
            # Set title and labels
            ax.set_title(f'{metric_name} by Training and Testing Size (Rank {rank})', fontsize=14)
            ax.set_xlabel('Test Size', fontsize=12)
            ax.set_ylabel('Training Size', fontsize=12)
            
            # Set ticks
            ax.set_xticks(np.arange(len(unique_test_sizes)))
            ax.set_yticks(np.arange(len(unique_train_sizes)))
            ax.set_xticklabels(unique_test_sizes, fontsize=10)
            ax.set_yticklabels(unique_train_sizes, fontsize=10)
            
            # Add value annotations
            for train_idx, train_size in enumerate(unique_train_sizes):
                for test_idx, test_size in enumerate(unique_test_sizes):
                    if (train_size, test_size) in train_test_values:
                        value = train_test_values[(train_size, test_size)]
                        text_color = 'white' if value > np.mean(metric_values) else 'black'
                        ax.text(test_idx, train_idx, f'{value:.3f}', 
                                ha='center', va='center', color=text_color, fontsize=9)
            
            # Add colorbar
            cbar = fig.colorbar(im)
            cbar.set_label(metric_name, fontsize=12)
            
            plt.tight_layout()
            fig.savefig(plot_dir / f"{metric_name.lower()}_heatmap_rank_{rank}.pdf")
            fig.savefig(plot_dir / f"{metric_name.lower()}_heatmap_rank_{rank}.png")
            plt.close(fig)
    
    # -------------------------------------------------------------------------
    # 2. Line plots showing how rank affects metrics for each train/test combination
    
    # Group data by train/test size combination
    train_test_groups = {}
    for i, (train_size, test_size) in enumerate(zip(train_data_sizes, test_data_sizes)):
        key = (train_size, test_size)
        if key not in train_test_groups:
            train_test_groups[key] = []
        train_test_groups[key].append(i)
    
    for metric_name, metric_values, ylim in [
        ('MAE', maes, None),
        ('RMSE', rmses, None),
        ('R²', r2s, (0, 1))
    ]:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f'Effect of Rank on {metric_name} by Data Size Configuration', fontsize=16)
        
        # Line plot for each train/test combination
        for (train_size, test_size), indices in train_test_groups.items():
            # Get ranks and metrics for this train/test combination
            group_ranks = [ranks[i] for i in indices]
            group_metrics = [metric_values[i] for i in indices]
            
            # Sort by rank
            sorted_indices = np.argsort(group_ranks)
            sorted_ranks = [group_ranks[i] for i in sorted_indices]
            sorted_metrics = [group_metrics[i] for i in sorted_indices]
            
            # Plot line
            label = f"Train: {train_size}, Test: {test_size}"
            marker = test_markers[test_size]
            color = train_colors[train_size]
            
            ax.plot(sorted_ranks, sorted_metrics, marker=marker, linestyle='-', 
                   color=color, linewidth=2, markersize=8, label=label)
            
            # Add text labels
            for i, (r, m) in enumerate(zip(sorted_ranks, sorted_metrics)):
                ax.text(r, m, f'{m:.3f}', ha='left', va='bottom', fontsize=8)
        
        ax.set_xlabel('Rank', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        
        # Create a custom legend with two parts: training size (marker) and test size (color)
        handles, labels = ax.get_legend_handles_labels()
        
        # Create legend
        if len(handles) > 10:  # If too many entries, create a compact legend
            ax.legend(fontsize=9, title="Data Configuration", loc='upper left', bbox_to_anchor=(1, 1))
        else:
            ax.legend(fontsize=10, title="Data Configuration")
        
        # Set x-axis to show only the integer ranks
        ax.set_xticks(sorted(unique_ranks))
        
        plt.tight_layout()
        fig.savefig(plot_dir / f"rank_vs_{metric_name.lower()}_by_train_test_size.pdf")
        fig.savefig(plot_dir / f"rank_vs_{metric_name.lower()}_by_train_test_size.png")
        plt.close(fig)
    
    # -------------------------------------------------------------------------
    # 3. Create comparison bar charts for selected combinations
    
    # Group by rank
    rank_groups = {}
    for i, rank in enumerate(ranks):
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(i)
    
    metrics_fig, axs = plt.subplots(2, 2, figsize=(16, 13))
    metrics_fig.suptitle('Model Performance Across Different Configurations', fontsize=20)
    
    # Prepare x-ticks - one group per rank
    n_ranks = len(unique_ranks)
    bar_width = 0.8 / len(train_test_groups)
    x_ticks = np.arange(n_ranks)
    
    # Counter for bar position within each rank group
    bar_positions = {}
    
    # Plot bars for each train/test combination
    for i, ((train_size, test_size), indices) in enumerate(train_test_groups.items()):
        # Get metrics for each rank with this train/test combination
        rank_to_metrics = {}
        for idx in indices:
            rank = ranks[idx]
            rank_to_metrics[rank] = {
                'mae': maes[idx],
                'rmse': rmses[idx],
                'r2': r2s[idx],
                'time': prediction_times[idx]
            }
        
        # Prepare data for all ranks (fill with NaN for missing combinations)
        rank_mae = [rank_to_metrics.get(r, {}).get('mae', np.nan) for r in unique_ranks]
        rank_rmse = [rank_to_metrics.get(r, {}).get('rmse', np.nan) for r in unique_ranks]
        rank_r2 = [rank_to_metrics.get(r, {}).get('r2', np.nan) for r in unique_ranks]
        rank_time = [rank_to_metrics.get(r, {}).get('time', np.nan) for r in unique_ranks]
        
        # Plot position for this train/test combination
        bar_pos = x_ticks + i * bar_width - 0.4 + bar_width/2
        
        # Label
        label = f"Train: {train_size}, Test: {test_size}"
        color = train_colors[train_size]
        
        # Plot bars
        bar1 = axs[0, 0].bar(bar_pos, rank_mae, width=bar_width, color=color, 
                         edgecolor='white', alpha=0.7, label=label)
        bar2 = axs[0, 1].bar(bar_pos, rank_rmse, width=bar_width, color=color, 
                         edgecolor='white', alpha=0.7, label=label)
        bar3 = axs[1, 0].bar(bar_pos, rank_r2, width=bar_width, color=color, 
                         edgecolor='white', alpha=0.7, label=label)
        bar4 = axs[1, 1].bar(bar_pos, rank_time, width=bar_width, color=color, 
                         edgecolor='white', alpha=0.7, label=label)
        
        # Add hatching based on test size
        for b1, b2, b3, b4 in zip(bar1, bar2, bar3, bar4):
            hatch = '/' if test_size == unique_test_sizes[0] else 'x' if test_size == unique_test_sizes[-1] else 'o'
            b1.set_hatch(hatch)
            b2.set_hatch(hatch)
            b3.set_hatch(hatch)
            b4.set_hatch(hatch)
        
        # Add value labels
        for j, (m1, m2, m3, m4) in enumerate(zip(rank_mae, rank_rmse, rank_r2, rank_time)):
            if not np.isnan(m1):
                axs[0, 0].text(bar_pos[j], m1 + 0.01, f'{m1:.3f}', ha='center', 
                              va='bottom', fontsize=9, rotation=45)
            if not np.isnan(m2):
                axs[0, 1].text(bar_pos[j], m2 + 0.01, f'{m2:.3f}', ha='center', 
                              va='bottom', fontsize=9, rotation=45)
            if not np.isnan(m3):
                axs[1, 0].text(bar_pos[j], m3 + 0.01, f'{m3:.3f}', ha='center', 
                              va='bottom', fontsize=9, rotation=45)
            if not np.isnan(m4):
                axs[1, 1].text(bar_pos[j], m4 + 0.01, f'{m4:.3f}', ha='center', 
                              va='bottom', fontsize=9, rotation=45)
    
    # Add titles and labels
    axs[0, 0].set_title('Mean Absolute Error (MAE)', fontsize=18)
    axs[0, 0].set_ylabel('MAE', fontsize=16)
    axs[0, 1].set_title('Root Mean Squared Error (RMSE)', fontsize=18)
    axs[0, 1].set_ylabel('RMSE', fontsize=16)
    axs[1, 0].set_title('R² Score', fontsize=18)
    axs[1, 0].set_ylabel('R²', fontsize=16)
    axs[1, 0].set_ylim(0, 1)
    axs[1, 1].set_title('Prediction Time (seconds)', fontsize=18)
    axs[1, 1].set_ylabel('Seconds', fontsize=16)
    
    # Set x-ticks
    for ax in axs.flatten():
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"Rank {r}" for r in unique_ranks], fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(alpha=0.3)
    
    # Create a custom legend to handle both color (train size) and hatching (test size)
    # Create patches for the train sizes (colors)
    train_patches = [plt.Rectangle((0,0), 1, 1, fc=train_colors[s], alpha=0.7, 
                                   label=f"Train size: {s}") for s in unique_train_sizes]
    
    # Create patches for the test sizes (hatches)
    test_patches = []
    hatch_patterns = ['/', 'x', 'o', '\\', '+', '*', '.']
    for i, s in enumerate(unique_test_sizes):
        pattern = hatch_patterns[i % len(hatch_patterns)]
        test_patches.append(plt.Rectangle((0,0), 1, 1, fill=False, hatch=pattern, 
                                          label=f"Test size: {s}"))
    
    # Add custom legend
    axs[0, 0].legend(handles=train_patches, title="Train Sizes", loc='upper right', fontsize=14, title_fontsize=14)
    axs[0, 1].legend(handles=test_patches, title="Test Sizes", loc='upper right', fontsize=14, title_fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    metrics_fig.savefig(plot_dir / "model_comparison_metrics.pdf")
    metrics_fig.savefig(plot_dir / "model_comparison_metrics.png")
    plt.close(metrics_fig)
    
    # -------------------------------------------------------------------------
    # 4. Create plots showing metrics per wall depth dimension
    for metric_name, metric_key in [
        ('MAE', 'mae_per_dim'),
        ('RMSE', 'rmse_per_dim'),
        ('R²', 'r2_per_dim')
    ]:
        # Make separate plots for each rank to avoid overcrowding
        for rank in unique_ranks:
            per_dim_fig, ax = plt.subplots(figsize=(12, 8))
            per_dim_fig.suptitle(f'{metric_name} per Wall Depth (Rank {rank})', fontsize=16)
            
            # Get model indices for this rank
            rank_indices = [i for i, r in enumerate(ranks) if r == rank]
            
            # Plot each train/test combination for this rank
            for idx in rank_indices:
                train_size = train_data_sizes[idx]
                test_size = test_data_sizes[idx]
                m = result_metrics[idx]
                
                values = m[metric_key]
                depths = range(len(values))
                
                marker = test_markers[test_size]
                color = train_colors[train_size]
                
                ax.plot(depths, values, marker=marker, linestyle='-', color=color, 
                       alpha=0.8, linewidth=2, markersize=5,
                       label=f"Train: {train_size}, Test: {test_size}")
            
            ax.set_xlabel('Wall Depth Index', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.grid(alpha=0.3)
            
            # Add legend with a reasonable size
            if len(rank_indices) > 6:
                ax.legend(fontsize=14, title="Data Configuration", 
                         loc='upper left', bbox_to_anchor=(1, 1))
            else:
                ax.legend(fontsize=14, title="Data Configuration")
            
            plt.tight_layout()
            per_dim_fig.savefig(plot_dir / f"comparison_rank_{rank}_{metric_key}.pdf")
            per_dim_fig.savefig(plot_dir / f"comparison_rank_{rank}_{metric_key}.png")
            plt.close(per_dim_fig)
    
    return {
        "ranks": ranks,
        "train_data_sizes": train_data_sizes,
        "test_data_sizes": test_data_sizes,
        "maes": maes,
        "rmses": rmses,
        "r2s": r2s,
        "prediction_times": prediction_times
    }

def evaluate_multitask_gpr(model_path, 
                           X_test_tensor, 
                           y_test,
                           X_train_tensor,
                           name_addition=""
                           ):

    model = DependentGPRModels()
    model.load(model_path)
    
    filename = Path(model_path).stem  # Remove .pkl extension
    parts = filename.split('_')
    n_epochs = int(parts[-4])
    lr = int(parts[-3])
    model_type = parts[0]
    training_points = int(parts[-2])
    try:
        rank = int(parts[-1])
    except:
        rank = ""
    # n_inducing_points = int(parts[-2])

    # create trained_models_plots directory if it doesn't exist
    Path("trained_models_plots").mkdir(parents=True, exist_ok=True)
    plot_dir = Path(r"trained_models_plots/{}_{}_surrogate_gpr_{}_{}_{}_{}".format(model_type, name_addition, n_epochs, lr, training_points, rank))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create predictions and save
    # save_predictions(model, X_test_tensor, plot_dir)

    # Load predictions
    try:    
        y_pred_df, y_var_df = load_predictions(plot_dir)
        y_pred = y_pred_df.to_numpy()
        y_var = y_var_df.to_numpy()
    except FileNotFoundError:
        print(f"No predictions found for {model_path}")
        save_predictions(model, X_test_tensor, plot_dir, model_path)
        y_pred_df, y_var_df = load_predictions(plot_dir)
        y_pred = y_pred_df.to_numpy()
        y_var = y_var_df.to_numpy()

    # # two subplots figure
    # fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    # ax1 = axes[0]
    # ax2 = axes[1]
    # ax1.hist(y_test[:,0])
    # ax2.hist(y_pred[:,0])
    # # plt.savefig(plot_dir / "histogram_comparison.png")
    # plt.show()

    # Plot results z
    plot(model, X_test_tensor, y_test, y_pred, y_var, X_train_tensor, plot_dir)

def evaluate_independent_gpr_models(model_path, 
                                    X_test_tensor, 
                                    y_test,
                                    ):
    model = IndependentGPRModels(output_dim=y_test.shape[1])
    model.load(model_path)

    # y_hat_test, var_test = model.predict(X_test_tensor)
    # y_hat_test = np.squeeze(y_hat_test)
    # var_test = np.squeeze(var_test)
    
    # Create plot directory if it doesn't exist
    # take model_path and extract n_epochs, lr, model_type, n_inducing_points
    n_epochs = int(model_path.split("_")[-2])
    lr = int(model_path.split("_")[-1].split(".")[0])   

    plot_dir = Path(r"figures/independent_surrogate_gpr_{}_{}".format(n_epochs, lr))
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create predictions and save
    try:
            # Load predictions
        y_pred_df, y_var_df = load_predictions(plot_dir)
        y_pred = y_pred_df.to_numpy()
        y_var = y_var_df.to_numpy()
    except FileNotFoundError:
        print(f"No predictions found for {model_path}")
        save_predictions(model, X_test_tensor, plot_dir)
        y_pred_df, y_var_df = load_predictions(plot_dir)
        y_pred = y_pred_df.to_numpy()
        y_var = y_var_df.to_numpy()

    # Plot results
    plot(model, X_test_tensor, y_test, y_pred, y_var, plot_dir)
    


class PrepareTrainingData:
    def __init__(self, X, y):
        y = [[item if item is not None else np.nan for item in row] for row in y]
        y = np.asarray(y)
        idx = np.where(~np.any(np.isnan(y), axis=-1))[0]

        self.X = X[idx]
        self.y = y[idx]

    def get_training_simple(self):
        return self.X, self.y
    
    def get_training_with_depth(self, depth_min=0, depth_max=-10):
        n_points_along_wall = self.y.shape[1]
        depths = np.linspace(depth_min, depth_max, n_points_along_wall)

        n_samples = self.X.shape[0]
        self.X = np.repeat(self.X, len(depths), axis=0)
        depths = np.repeat(depths, n_samples, axis=0)

        X_full = np.hstack((self.X, depths.reshape(-1, 1)))
        y_full = self.y.flatten()
        return X_full, y_full
    
    
if __name__ == "__main__":
     # Load data
    data_path = r"main/case_study_2025/data/srg_data_20250520_094244.csv"
    data_path = Path(Path(data_path).as_posix())
    data = pd.read_csv(data_path)

    filter_mask = np.zeros(data.shape[0], dtype=bool)

    n_lim = 150

    y = np.zeros((data.shape[0], n_lim))
    # plot columns disp_1 to disp_10 as boxplots
    # filter_mask = np.zeros(data.shape, dtype=bool)
    for i in range(1, n_lim+1):
        cur_data = data[f"disp_{i}"].values
        # ax1 = plt.subplot(1,2,1)
        q1 = np.percentile(cur_data, 25)
        q3 = np.percentile(cur_data, 75)
        iqr = q3 - q1
        
        whis = 1.5
        lower_bound = q1 - whis * iqr
        upper_bound = q3 + whis * iqr
        
        # Find the most extreme data points within the whisker bounds
        lower_whisker = np.min(cur_data[cur_data >= lower_bound])
        upper_whisker = np.max(cur_data[cur_data <= upper_bound])

        
        # remove outliers
        cur_filter_mask = (cur_data <= lower_whisker) | (cur_data >= upper_whisker)
        # cur_filter_mask = abs(y[:, i-1]) < 10
        filter_mask = filter_mask | cur_filter_mask
        y[:, i-1] = cur_data.copy()

    data = data[~filter_mask]


    # remove outliers from y
    y_og = y[~filter_mask]
    X_og = np.array([data["Klei_soilphi"].values, data["Klei_soilcohesion"].values, data["Klei_soilcurkb1"].values,
         data["Zand_soilphi"].values, data["Zand_soilcurkb1"].values,
         data["Zandlos_soilphi"].values, data["Zandlos_soilcurkb1"].values,
         data["Zandvast_soilphi"].values, data["Zandvast_soilcurkb1"].values,
         data["Wall_SheetPilingElementEI"].values])
    
    X_og = X_og.T
    nr_points = X_og.shape[0]

    target_nr_points = nr_points // 10
    # target_nr_points = 4
    # randomly select target_nr_points from X
    np.random.seed(42)
    selected_indices = np.random.choice(nr_points, target_nr_points, replace=False)
    X = X_og[selected_indices]
    y = y_og[selected_indices]
    
    prepare_training_data = PrepareTrainingData(X, y)
    X, y = prepare_training_data.get_training_simple()
    

    # keep every 10th column for y
    y = y[:, ::10]
    # print(f"y shape after removing 90% of the columns: {y.shape}")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)


    target_nr_points = nr_points // 10
    # target_nr_points = 50

    # randomly select target_nr_points from X
    np.random.seed(42)

    selected_indices = np.random.choice(nr_points, target_nr_points, replace=False)
    X = X_og[selected_indices]
    y = y_og[selected_indices]
    
    prepare_training_data = PrepareTrainingData(X, y)
    X, y = prepare_training_data.get_training_simple()
    
    # keep every 10th column for y
    y = y[:, ::10]
    # print(f"y shape after removing 90% of the columns: {y.shape}")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    added_value = 1000
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test + added_value, dtype=torch.float32)

    # Evaluate multitask modelx
    # model_path = r"results/multitask_gpr_surrogate_500_1_100_3186.pkl"
    # model_path = r"trained_models/multitask_gpr_surrogate_500_1_1592.pkl"
    
    for rank in [1]:
        for trainig_size in [1592]: #, 1592, 3186]:
            model_path = f"trained_models/multitask_gpr_surrogate_500_1_{trainig_size}_{rank}.pkl"
            evaluate_multitask_gpr(model_path, X_test_tensor, y_test_tensor, X_train_tensor, name_addition=f"Test_Data_{X_test.shape[0]}")


    # # # Compare all models with different ranks
    # comparison_dir = Path("trained_models_plots/model_comparison")
    # comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # result_metrics_paths = []
    # for rank in [1]: #,2,3,4]:
    #     metrics_path = Path(f"trained_models_plots/multitask_Test_Data_498_surrogate_gpr_1000_1_1991_{rank}/result_metrics.json")
    #     result_metrics_paths.append(metrics_path)
    #     metrics_path = Path(f"trained_models_plots/multitask_Test_Data_996_surrogate_gpr_1000_1_3983_{rank}/result_metrics.json")
    #     result_metrics_paths.append(metrics_path)

    # # Plot comparison of models with different ranks
    # if result_metrics_paths:
    #     plot_comparison_multitask_gpr_models(result_metrics_paths, comparison_dir)
    # #     print(f"Model comparison plots saved to {comparison_dir}")