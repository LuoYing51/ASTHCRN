import torch
import numpy as np

def inverse_normalize(data, scaler):
    original_shape = data.shape
    data_flat = data.reshape(-1, 1)
    denorm_flat = scaler.inverse_transform(data_flat)
    return denorm_flat.reshape(original_shape)

