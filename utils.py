# utils.py

import numpy as np

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

# Other utility functions can be added as needed
