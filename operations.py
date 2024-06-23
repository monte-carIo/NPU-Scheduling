# operations.py

import numpy as np

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
                curr = results[core_index][layer][0]
                batchnorm_result = batchnorm(curr, mean, variance, parameters["gamma"],
                                            parameters["beta"], parameters["eps"])
                # batchnorm_result = relu(batchnorm_result)
                results[core_index][layer] = (batchnorm_result, 
                                            results[core_index][layer][1], 
                                            results[core_index][layer][2])
    return results

# Other operations can be added as needed
