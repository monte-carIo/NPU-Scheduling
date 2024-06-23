import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict

class Node:
    def __init__(self, name, operation):
        self.name = name
        self.operation = operation
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

def create_computation_graph():
    # Define nodes with operation fusion
    input_node = Node("Input", None)
    fused_conv_bn_relu1 = Node("FusedConvBNReLU1", "Conv 3x3 -> BN -> ReLU")
    fused_conv_bn_relu2 = Node("FusedConvBNReLU2", "Conv 3x3 -> BN -> ReLU")
    fused_conv_bn_relu3 = Node("FusedConvBNReLU3", "Conv 3x3 -> BN -> ReLU")
    fused_conv_bn_relu4 = Node("FusedConvBNReLU4", "Conv 3x3 -> BN -> ReLU")
    residual_add1 = Node("ResidualAdd1", "Residual Add")
    residual_add2 = Node("ResidualAdd2", "Residual Add")
    concat = Node("Concat", "Concat")
    output_0 = Node("Output #0", None)
    conv_1x11 = Node("Conv 1x1_1", "Conv 1x1")
    conv_1x12 = Node("Conv 1x1_2", "Conv 1x1")
    output_1 = Node("Output #1", None)  

    # Define edges (data flow) ensuring order
    input_node.add_child(fused_conv_bn_relu1)
    input_node.add_child(residual_add1)
    fused_conv_bn_relu1.add_child(residual_add1)
    residual_add1.add_child(concat)
    residual_add1.add_child(fused_conv_bn_relu2)
    residual_add1.add_child(residual_add2)
    concat.add_child(conv_1x11)
    conv_1x11.add_child(output_0)
    fused_conv_bn_relu2.add_child(residual_add2)
    residual_add2.add_child(fused_conv_bn_relu3)
    residual_add2.add_child(fused_conv_bn_relu4)
    fused_conv_bn_relu4.add_child(concat)
    fused_conv_bn_relu3.add_child(conv_1x12)
    conv_1x12.add_child(output_1)

    # Construct computation graph
    computation_graph = {
        "nodes": [
            input_node,
            fused_conv_bn_relu1,
            fused_conv_bn_relu2,
            fused_conv_bn_relu3,
            fused_conv_bn_relu4,
            residual_add1,
            residual_add2,
            concat,
            output_0,
            conv_1x11,
            output_1,
            conv_1x12
        ]
    }
    return computation_graph

def visualize_graph(graph, levels):
    G = nx.DiGraph()

    # Add nodes and edges to the networkx graph
    for node in graph['nodes']:
        G.add_node(node.name, label=node.name, level=levels[node.name])
        for child in node.children:
            G.add_edge(node.name, child.name)

    pos = {}
    level_counts = defaultdict(int)
    for node, level in levels.items():
        pos[node] = (level_counts[level], -level)
        level_counts[level] += 1

    plt.figure(figsize=(12, 8))  # Set the figure size

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]['label'] for node in G.nodes()}, font_size=10)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='black', arrows=True)

    plt.title("Ordered Computation Graph with Operation Fusion", fontsize=16)
    plt.axis('off')  # Turn off the axis
    plt.show()

def topological_sort(graph):
    in_degree = defaultdict(int)
    adjacency_list = defaultdict(list)
    
    for node in graph['nodes']:
        for child in node.children:
            in_degree[child.name] += 1
            adjacency_list[node.name].append(child.name)
    
    zero_in_degree_queue = deque([node.name for node in graph['nodes'] if in_degree[node.name] == 0])
    
    L = [] 
    
    while zero_in_degree_queue:
        n = zero_in_degree_queue.popleft()
        L.append(n)
        
        for m in adjacency_list[n]:
            in_degree[m] -= 1
            if in_degree[m] == 0:
                zero_in_degree_queue.append(m)
    
    if len(L) != len(graph['nodes']):
        return "Error: The graph has at least one cycle"
    else:
        return L
    
def topological_sort_with_levels(graph):
    from collections import deque, defaultdict
    in_degree = defaultdict(int)
    adjacency_list = defaultdict(list)

    # Initialize in-degree and adjacency list
    for node in graph['nodes']:
        for child in node.children:
            in_degree[child.name] += 1
            adjacency_list[node.name].append(child.name)

    # Initialize the queue with nodes having zero in-degree
    zero_in_degree_queue = deque([node.name for node in graph['nodes'] if in_degree[node.name] == 0])
    
    # Result list and level dictionary
    L = []
    levels = {}
    current_level = 0

    # Process nodes level by level
    while zero_in_degree_queue:
        next_level_queue = deque()

        while zero_in_degree_queue:
            n = zero_in_degree_queue.popleft()
            L.append(n)
            levels[n] = current_level

            for m in adjacency_list[n]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    next_level_queue.append(m)

        # Move to the next level
        zero_in_degree_queue = next_level_queue
        current_level += 1

    if len(L) != len(graph['nodes']):
        return "Error: The graph has at least one cycle"
    else:
        return L, levels

# Generate the computation graph
computation_graph = create_computation_graph()

# Apply topological sort and print the result
# sorted_order = topological_sort(computation_graph)
# print("Topologically sorted order of the graph:")
# print(sorted_order)
# 
L, levels = topological_sort_with_levels(computation_graph)
# print("Topological Sort Order:", L)
print(levels)
visualize_graph(computation_graph, levels)