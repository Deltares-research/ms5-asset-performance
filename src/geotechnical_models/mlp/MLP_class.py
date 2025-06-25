import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):

        super().__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    

def inference(model, x, scaler_x, scaler_y, device=None):
    if device is None:
        device = next(model.parameters()).device
    x_scaled = scaler_x.transform(x)
    x_scaled_torch = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_hat_scaled = model(x_scaled_torch)
    y_hat_scaled = y_hat_scaled.detach().cpu().numpy()
    y_hat = scaler_y.inverse_transform(y_hat_scaled)
    return y_hat