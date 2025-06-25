import json
import pickle
from pathlib import Path
import numpy as np
import torch
import gpytorch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import gpytorch.settings
from torch.utils.data import DataLoader, TensorDataset
from tqdm import notebook, tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        # Linear mean
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.LinearMean(input_size=train_x.shape[-1]), num_tasks=num_tasks
        )
        # # Matern kernel
        # self.covar_module = gpytorch.kernels.MultitaskKernel(
        #     gpytorch.kernels.MaternKernel(nu=1.5), num_tasks=num_tasks, rank=rank
        # )
        #RBF kernel
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=rank
        )
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class SparseVariationaMultitaskGPModel(gpytorch.models.ApproximateGP):

    def __init__(self, inducing_points, num_tasks, num_latents=2):
        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_tasks, batch_shape=torch.Size([num_tasks])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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
        torch.set_num_threads(1)
        self.model = None
        self.likelihood = None
        self.scaler_x = None
        self.scaler_y = None

    def train(self, x_train, y_train, 
              scaler_x, scaler_y,
              n_epochs=500, lr=0.01,
              model_type="sparse", n_inducing_points=100, 
              path=None, device=None, rank=None):
        
        save_params = path is not None
        if save_params:
            if not isinstance(path, Path): path = Path(Path(path).as_posix())

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32)

        x_scaled = torch.tensor(scaler_x.fit_transform(x_train.numpy()), dtype=torch.float32)
        if model_type == "multitask" or model_type == "multitask-variational":
            y_i_scaled = torch.tensor(scaler_y.fit_transform(y_train.numpy()), dtype=torch.float32)
            num_tasks = y_i_scaled.shape[1]
        else:
            y_i_scaled = torch.tensor(scaler_y.fit_transform(y_train.numpy().reshape(-1, 1)).flatten(), dtype=torch.float32)

        self.scaler_x = scaler_x
        self.scaler_y = scaler_y

        if device is not None:
            x_scaled = x_scaled.to(device)
            y_i_scaled = y_i_scaled.to(device)
        
        if model_type == "exact":
            # Define the GP model and likelihood
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = ExactGPModel(x_scaled, y_i_scaled, self.likelihood)
        elif model_type == "sparse":
            # Define the GP model and likelihood
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = SparseGPModel(x_scaled, y_i_scaled, self.likelihood, nr_inducing_points=n_inducing_points)
        elif model_type == "multitask":
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
            if rank is not None:
                self.model = MultitaskGPModel(x_scaled, y_i_scaled, self.likelihood, num_tasks=num_tasks, rank=rank)
            else:
                self.model = MultitaskGPModel(x_scaled, y_i_scaled, self.likelihood, num_tasks=num_tasks)
        elif model_type == "multitask-variational":
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
            self.model = SparseVariationaMultitaskGPModel(inducing_points=x_scaled, num_tasks=num_tasks)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        if device is not None:
            self.model = self.model.to(device)
            self.likelihood = self.likelihood.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        self.likelihood.train()
        losses = []

        # Define the loss function (marginal log likelihood)
        if model_type == "multitask-variational":
            train_dataset = TensorDataset(x_scaled, y_i_scaled)
            train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=y_i_scaled.size(0))
            # We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
            # effective for VI.
            epochs_iter = notebook.tqdm(range(n_epochs), desc="Epoch")
            for i in epochs_iter:
                # Within each iteration, we will go over each minibatch of data
                minibatch_iter = notebook.tqdm(train_loader, desc="Minibatch", leave=False)
                for x_batch, y_batch in minibatch_iter:
                    optimizer.zero_grad()
                    output = self.model(x_batch)
                    loss = -mll(output, y_batch)
                    # epochs_iter.set_postfix(loss=loss.item())
                    minibatch_iter.set_postfix(loss=loss.item())
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())
        else:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
            # # Learning rate scheduler - reduce LR when loss plateaus
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer, mode='min', factor=0.5, patience=10, 
            #     verbose=True, threshold=1e-4
            # )
            print(f"Training {model_type} GP for {n_epochs} epochs")
            for epoch in tqdm(range(n_epochs)):
                optimizer.zero_grad()
                output = self.model(x_scaled)
                loss = -mll(output, y_i_scaled)
                loss.backward()          
                # print('Iter %d/%d - Loss: %.3f' % (epoch + 1, n_epochs, loss.item()))  
                optimizer.step()
                losses.append(loss.item())

                # Step the scheduler
                # scheduler.step(loss)
        
        # Set the model to evaluation mode
        self.model.eval()
        self.likelihood.eval()
        
        if save_params:
            with open(path / "model_params.pkl", 'wb') as f: 
                pickle.dump({
                    'model_state_dict': self.model.state_dict(),
                    'model': self.model,
                    'likelihood': self.likelihood,
                    'scaler_x': self.scaler_x,
                    'scaler_y': self.scaler_y,
                }, f)
            print(f"Saved model parameters at {path}")
        
        return np.array(losses), self.model.state_dict()
            
    def predict(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        X_scaled = torch.tensor(self.scaler_x.transform(x.numpy()), dtype=torch.float32)
        with torch.no_grad():
            model_output = self.model(X_scaled)
            observed_pred = self.likelihood(model_output)
                
            mean = observed_pred.mean
            variance = observed_pred.variance
                
            # Rescale predictions back to original scale
            mean_unscaled = self.scaler_y.inverse_transform(mean)
            variance_unscaled = variance.numpy() * self.scaler_y.scale_[0]**2
                
        return np.array(mean_unscaled), np.array(variance_unscaled)
    

def load_gpr_model(path):
    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    model = DependentGPRModels()
    with open(path, 'rb') as f:
        data = pickle.load(f)
        model.model = data['model']
        model.likelihood = data['likelihood']
        model.scaler_x = data['scaler_x']
        model.scaler_y = data['scaler_y']
    return model


def inference(model, x):
    return model.predict(x)[0]

        
# def plot_losses(losses_history, path):
#     if not isinstance(path, Path): path = Path(Path(path).as_posix())
    
#     fig = plt.figure(figsize=(12, 6))
#     for i, losses in enumerate(losses_history):
#         plt.plot(np.arange(1, len(losses)+1), losses, label=f'Output {i+1}')
    
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Negative Log Likelihood', fontsize=12)
#     plt.grid()
#     plt.legend()
#     plt.close()
    
#     fig.savefig(path)

# def run_full_parameter():
#     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     data_path = r"main/case_study_2025/train/data/srg_data_20250604_100638.csv"
#     # data_path = r"main/case_study_2025/train/data/srg_data_20250520_094244.csv"
#     data_path = Path(Path(data_path).as_posix())
#     data = pd.read_csv(data_path)

#     filter_mask = np.zeros(data.shape[0], dtype=bool)
#     n_lim = 150
#     y = np.zeros((data.shape[0], n_lim))
#     # plot columns disp_1 to disp_10 as boxplots
#     # filter_mask = np.zeros(data.shape, dtype=bool)
#     for i in range(1, n_lim+1):
#         cur_data = data[f"disp_{i}"].values
#         # ax1 = plt.subplot(1,2,1)
#         q1 = np.percentile(cur_data, 25)
#         q3 = np.percentile(cur_data, 75)
#         iqr = q3 - q1
        
#         whis = 1.5
#         lower_bound = q1 - whis * iqr
#         upper_bound = q3 + whis * iqr
        
#         # Find the most extreme data points within the whisker bounds
#         lower_whisker = np.min(cur_data[cur_data >= lower_bound])
#         upper_whisker = np.max(cur_data[cur_data <= upper_bound])

        
#         # remove outliers
#         cur_filter_mask = (cur_data <= lower_whisker) | (cur_data >= upper_whisker)
#         # cur_filter_mask = abs(y[:, i-1]) < 10
#         filter_mask = filter_mask | cur_filter_mask
#         y[:, i-1] = cur_data.copy()

#     data = data[~filter_mask]
#     # remove outliers from y
#     y = y[~filter_mask]
#     X = np.array([data["Klei_soilphi"].values, data["Klei_soilcohesion"].values, data["Klei_soilcurkb1"].values,
#          data["Zand_soilphi"].values, data["Zand_soilcurkb1"].values,
#          data["Zandlos_soilphi"].values, data["Zandlos_soilcurkb1"].values,
#          data["Zandvast_soilphi"].values, data["Zandvast_soilcurkb1"].values,
#          data["Wall_SheetPilingElementEI"].values, data["water_lvl"].values])
    
#     X = X.T
#     nr_points = X.shape[0]

#     divider = [2] # reduce the number of training points by a factor of divider_i
#     for divider_i in divider:
#         target_nr_points = nr_points // divider_i
#         # randomly select target_nr_points from X
#         np.random.seed(42)
#         selected_indices = np.random.choice(nr_points, target_nr_points, replace=False)
#         X = X[selected_indices]
#         y = y[selected_indices]
        
#         prepare_training_data = PrepareTrainingData(X, y)
#         X, y = prepare_training_data.get_training_simple()

#         # keep every 10th column for y
#         y = y[:, ::10]
#         # print(f"y shape after removing 90% of the columns: {y.shape}")

#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#         y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
#         X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#         y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
#         scaler_x = StandardScaler()
#         scaler_y = StandardScaler()
#         # scaler_x = MinMaxScaler(feature_range=(-1, 1))
#         # scaler_y = MinMaxScaler(feature_range=(-1, 1))


#         added_value = 0
#         # Convert to PyTorch tensors
        
#         # device = torch.device('mps')
#         # print(f"Using device: {device}")
#         # Create and train the model

        
#         n_epochs = 500
#         lr = 0.01
#         model_type = "multitask"
#         n_inducing_points = None
#         lr_for_str = int(lr*100)
#         nr_training_points = X_train_tensor.shape[0]

#         base_dir = Path(__file__).resolve().parent.parent
#         # mkdir trained_models
#         Path("trained_models").mkdir(parents=True, exist_ok=True)
#         for rank in [1]:
#             output_path = base_dir.parent / f"results/srg/gpr/surrogate_lr_{lr:.1e}_epochs_{n_epochs:d}_rank_{rank}"
#             output_path.mkdir(parents=True, exist_ok=True)
#         # for rank in [None]:
#             model = DependentGPRModels()
            
#             losses_history = model.train(
#                 X_train_tensor, 
#                 y_train_tensor, 
#                 scaler_x, scaler_y,
#                 n_epochs=n_epochs, lr=lr, n_inducing_points=n_inducing_points, model_type=model_type, # Reduced epochs for faster training
#                 path=output_path,
#                 rank=rank
#             )

#             y_hat_test, y_var_test = model.predict(X_test_tensor)
#             print(y_hat_test[:10])

#             print(f"MSE: {np.mean((y_hat_test - y_test)**2)}")

#             # plot_losses(losses_history, output_path / "losses.png")

# if __name__ == "__main__":
#     run_full_parameter()