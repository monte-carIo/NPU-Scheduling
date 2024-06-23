# main.py

import numpy as np
import torch
import torch.nn.functional as F
from utils import pad_input, divide_into_tiles, assign_tiles_to_cores
from operations import execute_on_cores

def initialize_parameters(grid_size, input_size, kernel_size, stride, padding, channels,
                          running_mean, running_var, gamma, beta, eps):
    return {
        "grid_size": grid_size,
        "input_size": input_size,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "channels": channels,
        "running_mean": running_mean,
        "running_var": running_var,
        "gamma": gamma,
        "beta": beta,
        "eps": eps
    }

def accumulate_results(results, input_size, tile_size, overlap, kernel_size, padding, stride):
    output_height = (input_size[0] + 2 * padding - kernel_size) // stride + 1 
    output_width = (input_size[1] + 2 * padding - kernel_size) // stride + 1
    output = np.zeros((output_height, output_width, input_size[2]))
    for core_result in results:
        for conv_result, i, j in core_result:
            output[i, j] += conv_result
    return output

def distribute_computations(parameters, input_data, kernel):
    input_data = pad_input(input_data, parameters["padding"])
    tile_size = parameters["kernel_size"]
    overlap = parameters["kernel_size"] - parameters["stride"]
    
    tiles = divide_into_tiles(input_data, tile_size, overlap)
    core_assignments = assign_tiles_to_cores(tiles, parameters["grid_size"])
    
    results = execute_on_cores(core_assignments, kernel, parameters)
    
    output = accumulate_results(results, parameters["input_size"], tile_size, overlap, parameters["kernel_size"], parameters["padding"], parameters["stride"])
    
    return output

# Example parameters
parameters = initialize_parameters(grid_size=(8, 10), input_size=(224, 224, 32),
                                kernel_size=3, stride=1, padding=1, channels=32,
                                running_mean = np.random.rand(32),
                                running_var = np.random.rand(32),
                                gamma = np.random.rand(32),
                                beta = np.random.rand(32),
                                eps = 1e-5)

# Example input data and kernel
input_data = np.random.rand(224, 224, 32)
kernel = np.random.rand(3, 3, 32, 32)

# Distribute computations and get output
output = distribute_computations(parameters, input_data, kernel)

# Print custom implementation output
print("Custom Implementation Output (First 10 values):")
print(output.flatten()[:10])

# Verify using PyTorch
input_tensor = torch.tensor(input_data).permute(2, 0, 1).unsqueeze(0)  # Convert to shape (1, C, H, W)
kernel_tensor = torch.tensor(kernel).permute(3, 2, 0, 1)  # Convert to shape (C_out, C_in, H, W)

conv_output_tensor = F.conv2d(input_tensor, kernel_tensor, stride=1, padding=1)
mean_tensor = torch.mean(conv_output_tensor, dim=(0, 2, 3))
variance_tensor = torch.var(conv_output_tensor, dim=(0, 2, 3))
gamma_tensor = torch.tensor(parameters["gamma"])
beta_tensor = torch.tensor(parameters["beta"])

bn_output_tensor = F.batch_norm(conv_output_tensor, mean_tensor, variance_tensor, gamma_tensor, beta_tensor, training=False, eps=parameters["eps"])
output_pytorch = bn_output_tensor.squeeze().permute(1, 2, 0).detach().numpy()  # Convert back to shape (H, W, C)

# Print PyTorch implementation output
print("PyTorch Implementation Output (First 10 values):")
print(output_pytorch.flatten()[:10])

# Compare the results
difference = np.abs(output - output_pytorch)
print("Difference (First 10 values):")
print(difference.flatten()[:10])

# Check if the difference is within an acceptable range
tolerance = 1e-4
if np.all(difference < tolerance):
    print("The outputs are similar within the tolerance level.")
else:
    print("The outputs differ beyond the tolerance level.")
