import numpy as np
import jax.numpy as jnp
from surrogate_mlp import NeuralNetwork
import pickle
from pathlib import Path
import json


data_path = r"results/sample_1000_unpooled.json"
data_path = Path(Path(data_path).as_posix())
with open(data_path, "r") as f: data = json.load(f)

y = data["displacement"]
y = [[item if item is not None else np.nan for item in row] for row in y]
y = np.asarray(y)

model = NeuralNetwork(y.shape[-1])
with open(r'results/mlp_surrogate.pkl', 'rb') as f: params = pickle.load(f)

def inference(model, x):
    return np.asarray(model.apply(params, jnp.asarray(x)))

