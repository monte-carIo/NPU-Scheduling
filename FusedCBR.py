import numpy as np
import torch
import torch.nn.functional as F

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

def pad_input(input_data, padding):
    if padding > 0:
        return np.pad(input_data, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    return input_data

def divide_into_tiles(input_data, tile_size, overlap):
    tiles = []
    step = tile_size - overlap
    for i in range(0, input_data.shape[0] - tile_size + 1, step):
        for j in range(0, input_data.shape[1] - tile_size + 1, step):
            tile = input_data[i:i+tile_size, j:j+tile_size]
            tiles.append((tile, i, j))
    return tiles

def assign_tiles_to_cores(tiles, grid_size):
    core_assignments = [[] for _ in range(grid_size[0] * grid_size[1])]
    for i, tile in enumerate(tiles):
        core_assignments[i % len(core_assignments)].append(tile)
    return core_assignments

def convolve(tile, kernel):
    return np.sum(tile * kernel, axis=(0, 1, 2))

def relu(tile):
    return np.maximum(tile, 0)

def batchnorm(tile, running_mean, running_var, gamma, beta, eps):
    batchnorm_result = gamma * (tile - running_mean) / np.sqrt(running_var + eps) + beta
    return batchnorm_result

def execute_on_cores(core_assignments, kernel, parameters):
    num_layers = len(core_assignments[0])
    results = [[] for _ in core_assignments]
    accum_mean = np.zeros(kernel.shape[-1], dtype=np.float32)
    accum_variance = np.zeros(kernel.shape[-1], dtype=np.float32)
    count = 0

    for layer in range(num_layers):
        for core_index, core_tiles in enumerate(core_assignments):
            if layer < len(core_tiles):
                tile, i, j = core_tiles[layer]
                conv_result = np.stack([convolve(tile, kernel[:, :, :, k]) for k in range(kernel.shape[-1])], axis=-1)
                results[core_index].append((conv_result, i, j))
                
                # Update mean and variance
                accum_mean += conv_result
                accum_variance += conv_result ** 2
                count += 1

    mean = accum_mean / count
    variance = accum_variance / count - mean ** 2
    
    for layer in range(num_layers):
        for core_index, core_tiles in enumerate(core_assignments):
            if layer < len(core_tiles):
                curr  = results[core_index][layer][0]
                batchnorm_result = batchnorm(curr, mean, variance, parameters["gamma"],
                                            parameters["beta"], parameters["eps"])
                # batchnorm_result = relu(batchnorm_result)
                results[core_index][layer] = (batchnorm_result, 
                                            results[core_index][layer][1], 
                                            results[core_index][layer][2])
    return results

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
