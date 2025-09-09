import os
import numpy as np
import pickle
import joblib
import torch
from pathlib import Path
import json
from typing import Optional
from scipy.interpolate import UnivariateSpline
from main.case_study_2025.train.surrogate.mlp_moment_train import MLP, MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class FoSCalculator:

    def __init__(self, wall_props, model_path, scaler_x_path, scaler_y_path):

        if not isinstance(model_path, Path): model_path = Path(Path(model_path).as_posix())
        self.model = MLP(
            input_dim=11,
            hidden_dims=[1024, 512, 256, 128, 64, 32],
            output_dim=1
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        if not isinstance(scaler_x_path, Path): scaler_x_path = Path(Path(scaler_x_path).as_posix())
        self.scaler_x = joblib.load(scaler_x_path)

        if not isinstance(scaler_y_path, Path): scaler_y_path = Path(Path(scaler_y_path).as_posix())
        self.scaler_y = joblib.load(scaler_y_path)

        self.wall_props = wall_props

    def inference(self, X):
        X_scaled = self.scaler_x.transform(X.cpu().numpy())
        X_scaled_tensor = torch.from_numpy(X_scaled).float()
        y_scaled = self.model(X_scaled_tensor)
        y_scaled = y_scaled.detach().numpy()
        y = self.scaler_y.inverse_transform(y_scaled)
        return y

    def moments(self, X):

        # EI, _, wall_locs, monitoring_locs = self.wall_props
        # _, keep_idx = np.unique(wall_locs, return_index=True)
        # keep_idx = np.sort(keep_idx)
        # wall_locs = wall_locs[keep_idx]

        moments = self.inference(X)

        return moments


if __name__ == "__main__":

    pass

