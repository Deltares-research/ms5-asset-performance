import json
import pickle
from pathlib import Path
import numpy as np
from typing import Optional, Tuple
import torch
import gpytorch
from torch import Tensor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class IndependentGPRModels:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.models = []
        self.likelihoods = []
        self.scalers_x = []
        self.scalers_y = []

    def train(self, x_train, y_train, n_epochs=500, lr=0.01, path=None):
        save_params = path is not None
        if save_params:
            if not isinstance(path, Path): path = Path(Path(path).as_posix())

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32)

        # Create separate GP models for each output dimension
        losses_history = []
        for i in range(self.output_dim):
            # Scale the inputs and outputs
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            
            x_scaled = torch.tensor(scaler_x.fit_transform(x_train.numpy()), dtype=torch.float32)
            y_i_scaled = torch.tensor(scaler_y.fit_transform(y_train[:, i].reshape(-1, 1)).flatten(), dtype=torch.float32)
            
            self.scalers_x.append(scaler_x)
            self.scalers_y.append(scaler_y)
            
            # Define the GP model and likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(x_scaled, y_i_scaled, likelihood)
            
            
            # Use the Adam optimizer with a learning rate scheduler
            optimizer = torch.optim.Adam([
                {'params': model.covar_module.parameters()},
                {'params': model.mean_module.parameters()},
                {'params': model.likelihood.parameters()},
            ], lr=lr)
            
            # Learning rate scheduler - reduce LR when loss plateaus
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, 
                verbose=True, threshold=1e-4
            )
            
            # Define the loss function (marginal log likelihood)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            # Train the model
            model.train()
            likelihood.train()
            losses = []
            
            print(f"Training GP for output dimension {i+1}/{self.output_dim}")
            for epoch in tqdm(range(n_epochs)):
                optimizer.zero_grad()
                output = model(x_scaled)
                loss = -mll(output, y_i_scaled)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                # Step the scheduler
                scheduler.step(loss)
            
            losses_history.append(losses)
            
            # Set the model to evaluation mode
            model.eval()
            likelihood.eval()
            
            self.models.append(model)
            self.likelihoods.append(likelihood)
        
        if save_params:
            with open(path, 'wb') as f: 
                pickle.dump({
                    'models': self.models,
                    'likelihoods': self.likelihoods,
                    'scalers_x': self.scalers_x,
                    'scalers_y': self.scalers_y
                }, f)
            print(f"Saved model parameters at {path}")
        
        return losses_history

    def load(self, path):
        if not isinstance(path, Path): path = Path(Path(path).as_posix())
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.likelihoods = data['likelihoods']
            self.scalers_x = data['scalers_x']
            self.scalers_y = data['scalers_y']
            
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        predictions = []
        variances = []
        
        with torch.no_grad():
            for i in range(self.output_dim):
                model = self.models[i]
                likelihood = self.likelihoods[i]
                
                x_scaled = torch.tensor(self.scalers_x[i].transform(x.numpy()), dtype=torch.float32)
                
                # Get predictions from model
                observed_pred = likelihood(model(x_scaled))
                mean = observed_pred.mean
                variance = observed_pred.variance
                
                # Rescale predictions back to original scale
                mean_unscaled = self.scalers_y[i].inverse_transform(mean.reshape(-1, 1)).flatten()
                variance_unscaled = variance.numpy() * self.scalers_y[i].scale_[0]**2
                
                predictions.append(mean_unscaled)
                variances.append(variance_unscaled)
        
        return np.stack(predictions, axis=1), np.stack(variances, axis=1)


def inference(model, x):
    return model.predict(x)[0]


def plot_predictions_with_uncertainty(model, x_train, x_test, y_train, y_test, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
    y_hat_train, var_train = model.predict(x_train)
    y_hat_test, var_test = model.predict(x_test)
    
    figs = []
    zipped = zip(y_train.T, y_hat_train.T, y_test.T, y_hat_test.T, var_test.T)
    for i_point, (y_t_train, y_p_train, y_t_test, y_p_test, var_test_i) in enumerate(zipped):
        fig = plt.figure()
        fig.suptitle(f"Point #{i_point+1:d} along wall", fontsize=14)
        
        # Training data
        plt.scatter(y_t_train, y_p_train, marker='x', c='b', label="Train")
        
        # Test data with uncertainty
        plt.errorbar(y_t_test, y_p_test, yerr=2*np.sqrt(var_test_i), fmt='x', c='r', 
                     ecolor='r', alpha=0.3, label="Test (±2σ)")
        
        plt.axline((0, 0), slope=1, c='k')
        plt.plot([y_t_train.min(), y_t_train.min()], [y_t_train.max(), y_t_train.max()], c='k')
        plt.xlabel('Observation', fontsize=12)
        plt.ylabel('Prediction', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        plt.close()
        figs.append(fig)
    
    pp = PdfPages(path)
    [pp.savefig(fig) for fig in figs]
    pp.close()


def plot_wall(model, x_train, y_train, x_test, y_test, path, depth_lims=[0, -10]):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
    x_data = [x_train, x_test]
    y_data = [y_train, y_test]

    fig = plt.figure()
    for i in range(len(x_data)):
        # create subplot
        ax = fig.add_subplot(1, len(x_data), i+1)
        y_hat, var = model.predict(x_data[i])
        residuals = y_data[i] - y_hat
        # st_residuals = residuals / np.abs(y_data[i]+1e-8) * 100

        depths = np.linspace(depth_lims[0], depth_lims[1], len(model.models))
        
        # Standard deviation from variance
        # std_dev = np.sqrt(var)
        
        for i, r in enumerate(residuals):
            ax.plot(r, depths, c='b', alpha=0.3)
        
        # Plot average error with uncertainty bands
        mean_error = np.mean(residuals, axis=0)
        std_error = np.std(residuals, axis=0)
        ax.plot(mean_error, depths, c='r', linewidth=2, label='Mean Error')
        ax.fill_betweenx(depths, 
                        mean_error - 2*std_error, 
                        mean_error + 2*std_error, 
                        color='r', alpha=0.2, label='±2σ')
        
        ax.set_xlabel('Error [mm]', fontsize=12)
        ax.set_ylabel('Depth [mm]', fontsize=12)
        ax.set_title(f"Error along wall for {'Training' if i == 0 else 'Test'} data", fontsize=14)
        ax.grid()
        ax.legend()
    
    pp = PdfPages(path)
    pp.savefig(fig)
    pp.close()
    
    # Save as PNG as well
    png_path = path.with_suffix('.png')
    plt.savefig(png_path)


def plot_variables(model, x, y, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
    y_hat, var = model.predict(x)
    residuals = y - y_hat
    st_residuals = residuals / np.abs(y+1e-8) * 100
    
    figs = []
    for i_point, r in enumerate(st_residuals.T):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
        fig.suptitle(f"Point #{i_point+1:d} along wall", fontsize=14)
        
        sc = axs[0].scatter(x[:, 0], r, c=x[:, 1])
        cbar = fig.colorbar(sc, ax=axs[0])
        cbar.set_label("Cohesion [kPa]", fontsize=10, labelpad=10)
        axs[0].set_xlabel('Phi [deg]', fontsize=12)
        axs[0].set_ylabel('Error [%]', fontsize=12)
        
        sc = axs[1].scatter(x[:, 1], r, c=x[:, 0])
        cbar = fig.colorbar(sc, ax=axs[1])
        cbar.set_label("Phi [deg]", fontsize=10, labelpad=10)
        axs[1].set_xlabel('Cohesion [kPa]', fontsize=12)
        axs[1].set_ylabel('Error [%]', fontsize=12)
        
        axs[0].set_ylim(residuals.min(), residuals.max())
        axs[1].set_ylim(residuals.min(), residuals.max())
        axs[0].grid()
        axs[1].grid()
        plt.close()
        figs.append(fig)
        
    pp = PdfPages(path)
    [pp.savefig(fig) for fig in figs]
    pp.close()


def plot_losses(losses_history, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
    fig = plt.figure(figsize=(12, 6))
    for i, losses in enumerate(losses_history):
        plt.plot(np.arange(1, len(losses)+1), losses, label=f'Output {i+1}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Negative Log Likelihood', fontsize=12)
    plt.grid()
    plt.legend()
    plt.close()
    
    fig.savefig(path)


def plot(model, x_train, x_test, y_train, y_test, path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
    # plot_predictions_with_uncertainty(model, x_train, x_test, y_train, y_test, path/"predictions.pdf")
    plot_wall(model, x_train, y_train, x_test, y_test, path/"wall.pdf")
    # plot_variables(model, x_train, y_train, path/"variables.pdf")


def compute_metrics(y_true, y_pred):
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
    
    # Overall metrics
    overall_mse = np.mean(mse_per_dim)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = np.mean(mae_per_dim)
    overall_r2 = np.mean(r2_per_dim)
    
    return {
        'mse_per_dim': mse_per_dim,
        'rmse_per_dim': rmse_per_dim,
        'mae_per_dim': mae_per_dim,
        'r2_per_dim': r2_per_dim,
        'overall_mse': overall_mse,
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'overall_r2': overall_r2
    }

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = r"results/sample_3000_unpooled.json"
    data_path = Path(Path(data_path).as_posix())
    with open(data_path, "r") as f: data = json.load(f)
    
    y = data["displacement"]
    y = [[item if item is not None else np.nan for item in row] for row in y]
    y = np.asarray(y)
    idx = np.where(~np.any(np.isnan(y), axis=-1))[0]
    
    X = (data["Klei_soilphi"], data["Klei_soilcohesion"], data["Zand_soilphi"], data["Wall_SheetPilingElementEI"])
    X = tuple([np.asarray(item) for item in X])
    X = np.stack(X, axis=-1)
    
    X = X[idx]
    y = y[idx]
    
    # keep every 10th column for y
    y = y[:, ::10]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create and train the model
    model = IndependentGPRModels(output_dim=y.shape[1])
    
    # Either train a new model or load a previously trained one
    train_new_model = True  # Set to True to train a new model, False to load existing
    
    # Check if the model file exists
    # model_path = Path(r'results/gpr_surrogate.pkl')
    # if not model_path.exists():
    #     print(f"Model file {model_path} does not exist, training a new model.")
    #     train_new_model = True
    
    n_epochs = 100
    lr = 0.01
    lr_for_str = int(lr*100)

    if train_new_model:
        # For GPR, we can use fewer epochs than for neural networks
        losses_history = model.train(
            X_train_tensor, y_train_tensor, 
            n_epochs=n_epochs, lr=lr,  # Reduced epochs for faster training
            path=r'results/independent_gpr_surrogate_{}_{}.pkl'.format(n_epochs, lr_for_str)
        )
        Path(r"figures/independent_surrogate_gpr_{}_{}".format(n_epochs, lr_for_str)).mkdir(parents=True, exist_ok=True)
        plot_losses(losses_history, r"figures/independent_surrogate_gpr_{}_{}/losses.png".format(n_epochs, lr_for_str))
    else:
        model.load(r'results/independent_gpr_surrogate_{}_{}.pkl'.format(n_epochs, (lr*100)))
    
    # Create plot directory if it doesn't exist
    plot_dir = Path(r"figures/independent_surrogate_gpr_{}_{}".format(n_epochs, lr_for_str))
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    plot(model, X_train_tensor, X_test_tensor, y_train, y_test, plot_dir)
    
    # # Generate predictions for all data and compute correlation matrix of residuals
    # y_hat = inference(model, torch.tensor(X, dtype=torch.float32))
    # residuals = y - y_hat
    # corr_mat = np.corrcoef(residuals.T)
    # np.save(r"results/gpr_corr_mat.npy", corr_mat)