# NPU Scheduling Problem Solving

## Introduction

This project aims to solve the scheduling problem for Neural Processing Units (NPU) by efficiently distributing computations across a grid of cores. The provided solution leverages a custom computation graph and an optimized dataflow algorithm to enhance the performance of convolutional neural network (CNN) operations on NPUs.

## Problem Description

We are given an NPU consisting of an 8x10 grid of cores. Each core can perform batched MAC (Multiply-Accumulate) operations and elementwise operations, such as 32x32 matrix multiplication, elementwise multiplication/addition, and nonlinearities. Data flows through rows or columns, routed via algorithms, entering through the bottommost cores and exiting through the same cores. Each core has a local cache of 1MB, and only one operation can be run across all cores simultaneously.

## Algorithm Overview

### Creating the Computation Graph

1. **Node Class**: Defines a node in the computation graph with a name, operation, and child nodes.
2. **Graph Construction**: Nodes representing fused convolution, batch normalization, and ReLU operations (FusedCBR) are created and connected to form the computation graph.

### Topological Sorting

Using Kahn's algorithm, we generate a topologically sorted order of nodes with additional handling for graph levels, representing the order of execution.

### Distributing Computations

1. **Tiles Dividing**: The input data is divided into tiles based on kernel size and channels, ensuring minimal memory consumption.
2. **Cores Assignments**: Tiles are assigned to cores in a round-robin manner to maximize parallelism.
3. **Dataflow During Runtime**: Data flows through cores according to the assigned tiles, leveraging local cache memory for efficient computation.

## Implementation

### Key Functions

- **create_computation_graph()**: Constructs the computation graph.
- **topological_sort_with_levels(graph)**: Performs topological sorting on the computation graph.
- **divide_into_tiles(input_data, tile_size, overlap)**: Divides input data into tiles for processing.
- **assign_tiles_to_cores(tiles, grid_size)**: Assigns tiles to NPU cores.
- **execute_on_cores(core_assignments, kernel, parameters)**: Executes computations on the assigned cores.
- **accumulate_results(results, input_size, tile_size, overlap, kernel_size, padding, stride)**: Accumulates results from cores to form the final output.

### Example Usage

```python
parameters = initialize_parameters(grid_size=(8, 10), input_size=(224, 224, 32),
                                   kernel_size=3, stride=1, padding=1, channels=32,
                                   running_mean=np.random.rand(32),
                                   running_var=np.random.rand(32),
                                   gamma=np.random.rand(32),
                                   beta=np.random.rand(32),
                                   eps=1e-5)

input_data = np.random.rand(224, 224, 32)
kernel = np.random.rand(3, 3, 32, 32)

output = distribute_computations(parameters, input_data, kernel)

print("Output (First 10 values):")
print(output.flatten()[:10])
```

## Future Work

Further optimizations can reduce memory consumption caused by overlapping tiles and improve data access latency by exploring new approaches to tile indexing and memory management.

## References

- [GitHub Repository](https://github.com/monte-carIo/NPU-Scheduling)
- [Notion Report](https://www.notion.so/10b0d1b688974ccf94ad658ce282486d?pvs=25)

This README provides a high-level overview of the NPU Scheduling Problem Solving project, its key algorithms, and implementation details. For a complete understanding, refer to the provided code and documentation.