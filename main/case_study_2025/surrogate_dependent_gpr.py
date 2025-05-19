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
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=2
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
    
class SparseGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, nr_inducing_points=500):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # randomly select nr_inducing_points from train_x
        inducing_idx = np.random.choice(train_x.shape[0], nr_inducing_points, replace=False)
        print(f"shape of train_x: {train_x.shape}")
        print(f"shape of inducing_idx: {inducing_idx.shape}")
        print(f"shape of train_x[inducing_idx, :]: {train_x[inducing_idx, :].shape}")
        
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, 
                                                                 inducing_points=train_x[inducing_idx, :].clone(), 
                                                                 likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DepthAwareGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, depth_idx=-1, nr_inducing_points=500):
        super(DepthAwareGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.depth_idx = depth_idx
        
        # Randomly select inducing points
        inducing_idx = np.random.choice(train_x.shape[0], nr_inducing_points, replace=False)
        inducing_points = train_x[inducing_idx].clone()
        
        # Soil parameters kernel (excluding depth)
        self.soil_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        # Depth-specific kernel with potentially smaller lengthscale to capture local continuity
        self.depth_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale_prior=gpytorch.priors.GammaPrior(3, 6))
        )
        
        # Combine kernels using a ProductKernel
        base_kernel = self.soil_kernel * self.depth_kernel
        
        # Sparse GP approximation using inducing points
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            base_kernel, inducing_points=inducing_points, likelihood=likelihood
        )
    
    def forward(self, x):
        mean = self.mean_module(x)
        
        # Extract soil parameters and depth
        if self.depth_idx == -1:
            soil_x = x[:, :-1]
            depth_x = x[:, [-1]]
        else:
            # Create mask for all columns except depth
            mask = torch.ones(x.shape[1], dtype=torch.bool)
            mask[self.depth_idx] = False
            soil_x = x[:, mask]
            depth_x = x[:, [self.depth_idx]]
        
        soil_covar = self.soil_kernel(soil_x)
        depth_covar = self.depth_kernel(depth_x)
        
        covar = soil_covar * depth_covar
        # Apply the kernel directly to x (the product kernel handles dimension separation)
        # covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    

class DependentGPRModels:
    def __init__(self):
        self.models = []
        self.likelihoods = []
        self.scalers_x = []
        self.scalers_y = []

    def train(self, x_train, y_train, n_epochs=500, lr=0.01, model_type="sparse", n_inducing_points=100, path=None):
        save_params = path is not None
        if save_params:
            if not isinstance(path, Path): path = Path(Path(path).as_posix())

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32)

        # Create separate GP models for each output dimension
        losses_history = []
        # for i in range(self.output_dim):
            # Scale the inputs and outputs
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        
        # print(f"x_train.shape: {x_train.numpy().shape}")
        # print(f"y_train.shape: {y_train.numpy().reshape(-1, 1).shape}")
        x_scaled = torch.tensor(scaler_x.fit_transform(x_train.numpy()), dtype=torch.float32)
        if model_type == "multitask":
            y_i_scaled = torch.tensor(scaler_y.fit_transform(y_train.numpy()), dtype=torch.float32)
            num_tasks = y_i_scaled.shape[1]
        else:
            y_i_scaled = torch.tensor(scaler_y.fit_transform(y_train.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32)

        
        self.scalers_x.append(scaler_x)
        self.scalers_y.append(scaler_y)
        
        if model_type == "exact":
            # Define the GP model and likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = ExactGPModel(x_scaled, y_i_scaled, likelihood)
        elif model_type == "sparse":
            # Define the GP model and likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            model = SparseGPModel(x_scaled, y_i_scaled, likelihood, nr_inducing_points=n_inducing_points)
        elif model_type == "multitask":
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
            model = MultitaskGPModel(x_scaled, y_i_scaled, likelihood, num_tasks=num_tasks)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # Use the Adam optimizer with a learning rate scheduler
        # optimizer = torch.optim.Adam([
        #     {'params': model.covar_module.parameters()},
        #     {'params': model.mean_module.parameters()},
        #     {'params': model.likelihood.parameters()},
        # ], lr=lr)
            
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

        print(f"Training sparse GP for {n_epochs} epochs")
        for epoch in tqdm(range(n_epochs)):
            optimizer.zero_grad()
            output = model(x_scaled)
            loss = -mll(output, y_i_scaled)
            loss.backward()

            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #   epoch + 1, n_epochs, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item()
            # ))
            
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
            for i in range(len(self.models)):
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
    
    fig = plt.figure()
    fig.suptitle(f"Training and test data", fontsize=14)
    
    # Make sure all arrays are 1D for plotting
    y_train_flat = y_train.numpy().flatten() if torch.is_tensor(y_train) else y_train.flatten()
    y_hat_train_flat = y_hat_train.flatten()
    
    y_test_flat = y_test.numpy().flatten() if torch.is_tensor(y_test) else y_test.flatten()
    y_hat_test_flat = y_hat_test.flatten()
    y_error = 2*np.sqrt(var_test.flatten())
    
    # Training data
    plt.scatter(y_train_flat, y_hat_train_flat, marker='x', c='b', label="Train")
    
    # Test data with uncertainty
    plt.errorbar(y_test_flat, y_hat_test_flat, yerr=y_error, fmt='x', c='r', 
                 ecolor='r', alpha=0.3, label="Test (±2σ)")
    
    plt.axline((0, 0), slope=1, c='k')
    plt.xlabel('Observation', fontsize=12)
    plt.ylabel('Prediction', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()
    
    pp = PdfPages(path)
    pp.savefig(fig)
    pp.close()


def plot_wall(model, x_train, y_train, x_test, y_test, path, depth_lims=[0, -10]):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
    # Set up figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    
    # Get data for train and test
    x_data = [x_train, x_test]
    y_data = [y_train, y_test]
    titles = ["Training", "Test"]
    
    # Use a single depth scale for both plots
    depths = np.linspace(depth_lims[0], depth_lims[1], 10)  # Simplified depth representation
    
    # Process each dataset (train/test)
    for idx, (x, y, title, ax) in enumerate(zip(x_data, y_data, titles, axs)):
        # Get predictions
        y_hat, var = model.predict(x)
        
        # Convert tensors to numpy if needed
        if torch.is_tensor(y):
            y = y.numpy()
    
        # Calculate residuals
        residuals = y - y_hat
    
        # Get statistics
        mean_error = np.mean(residuals, axis=0)
        std_error = np.std(residuals, axis=0)
        
        # Only plot a sample of individual residuals to reduce clutter
        max_lines = min(20, len(residuals))
        indices = np.linspace(0, len(residuals) - 1, max_lines, dtype=int)
        for i in indices:
            ax.plot(residuals[i], x[:, -1], c='b', alpha=0.1)
        
        # Plot mean with confidence band
        ax.plot(mean_error, depths, c='r', linewidth=2, label='Mean Error')
        ax.fill_betweenx(depths, 
                      mean_error - 2*std_error, 
                      mean_error + 2*std_error, 
                      color='r', alpha=0.2, label='±2σ')
        
        # Labels and styling
        ax.set_xlabel('Error [mm]', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Depth [m]', fontsize=12)
        ax.set_title(f"{title} data", fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save as PDF and PNG
    pp = PdfPages(path)
    pp.savefig(fig)
    pp.close()
    
    png_path = path.with_suffix('.png')
    plt.savefig(png_path)
    plt.close(fig)


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
        cbar = fig.colorbar(sc, ax=axxs[0])
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
    # plot_wall(model, x_train, y_train, x_test, y_test, path/"wall.pdf")
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

class PrepareTrainingData:
    def __init__(self, X, y):
        y = [[item if item is not None else np.nan for item in row] for row in y]
        y = np.asarray(y)
        idx = np.where(~np.any(np.isnan(y), axis=-1))[0]

        X = tuple([np.asarray(item) for item in X])
        X = np.stack(X, axis=-1)
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
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = r"results/sample_3000_unpooled.json"
    data_path = Path(Path(data_path).as_posix())
    with open(data_path, "r") as f: data = json.load(f)
    
    y = data["displacement"]
    X = (data["Klei_soilphi"], data["Klei_soilcohesion"], data["Zand_soilphi"], data["Wall_SheetPilingElementEI"])

    prepare_training_data = PrepareTrainingData(X, y)
    X, y = prepare_training_data.get_training_simple()

    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    # keep every 10th column for y
    y = y[:, ::10]
    print(f"y.shape after: {y.shape}")
    # X, y = prepare_training_data.get_training_with_depth()
    

    # take every 10th point
    # step = 100
    # X = X[::step, :]
    # y = y[::step]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(f"X_train.shape: {X_train.shape}")
    # print(f"X_test.shape: {X_test.shape}")
    # print(f"y_train.shape: {y_train.shape}")
    # print(f"y_test.shape: {y_test.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create and train the model
    model = DependentGPRModels()
    
    # Either train a new model or load a previously trained one
    train_new_model = True  # Set to True to train a new model, False to load existing
    
    # # Check if the model file exists
    # model_path = Path(r'results/gpr_surrogate.pkl')
    # if not model_path.exists():
    #     print(f"Model file {model_path} does not exist, training a new model.")
    #     train_new_model = True

    n_epochs = 100
    lr = 0.01
    model_type = "multitask"
    n_inducing_points = 100

    if train_new_model:
        # For GPR, we can use fewer epochs than for neural networks
        losses_history = model.train(
            X_train_tensor, 
            y_train_tensor, 
            n_epochs=n_epochs, lr=lr, n_inducing_points=n_inducing_points, model_type=model_type, # Reduced epochs for faster training
            path=r'results/gpr_surrogate_{}_{}_{}_{}.pkl'.format(n_epochs, lr, model_type, n_inducing_points)
        )
        plot_losses(losses_history, r"figures/surrogate_gpr/losses.png")
    else:
        model.load(r'results/gpr_surrogate_{}_{}_{}_{}.pkl'.format(n_epochs, lr, model_type, n_inducing_points))
    
    # Create plot directory if it doesn't exist
    plot_dir = Path(r"figures/surrogate_gpr_{}_{}_{}_{}".format(n_epochs, lr, model_type, n_inducing_points))
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    plot(model, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, plot_dir)
    
    # Generate predictions for all data and compute correlation matrix of residuals
    # y_hat = inference(model, torch.tensor(X, dtype=torch.float32))
    # residuals = y - y_hat
    # corr_mat = np.corrcoef(residuals.T)
    # np.save(r"results/gpr_corr_mat.npy", corr_mat) 