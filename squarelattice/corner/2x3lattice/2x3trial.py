import re

from pprint import pprint
from pyqubo import Array, Binary
import neal
import matplotlib.pyplot as plt
import os
import shutil
import time
import psutil

# Delete the output directory if it exists and create it again
output_dir = "pic"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Delete the information file if it exists and create a new one
info_file = "2x3infor.txt"
if os.path.exists(info_file):
    os.remove(info_file)

# Define the lattice size
num_rows = 2
num_cols = 3

# Define binary variables for nodes and bonds using Array to make scaling easier
nodes = Array.create('x', shape=(num_rows, num_cols), vartype='BINARY')
bonds = {}

# Define bond variables (upper layer, lower layer, and between layers)
for row in range(num_rows):
    for col in range(num_cols - 1):
        bonds[f'x_{row}{col}_to_{row}{col+1}'] = Binary(f'x_{row}{col}_to_{row}{col+1}')

# Define bonds between upper and lower layers
for col in range(num_cols):
    bonds[f'x_0{col}_to_1{col}'] = Binary(f'x_0{col}_to_1{col}')

# Define the Hamiltonian for the nodes
N = 5 # Set the total number of active nodes
L = 4  # Set the total number of active bonds
A_mono = 1
A_bond_all = 10
A_corner=1
A_diag = 1
A_bond = 1
A_adjacent = 1
# basic cons
H_node = A_mono * (sum([nodes[0, col] for col in range(num_cols)]) + sum([nodes[1, col] for col in range(num_cols)]) - N) ** 2
H_bond_all = A_bond * (sum(bonds.values()) - L) ** 2

# Define each bond Hamiltonian term separately to ensure compatibility
H_bond = 0

# Add bonds within the same layer
for row in range(num_rows):
    for col in range(num_cols - 1):
        bond_var = bonds[f'x_{row}{col}_to_{row}{col+1}']
        H_bond += bond_var * (1 - nodes[row, col]) + bond_var * (1 - nodes[row, col + 1])

# Add bonds between upper and lower layers
for col in range(num_cols):
    bond_var = bonds[f'x_0{col}_to_1{col}']
    H_bond += bond_var * (1 - nodes[0, col]) + bond_var * (1 - nodes[1, col])

# 正则表达式来提取坐标
def extract_points(key):
    return tuple(re.findall(r'\d{2}', key))

# 检查是否可以形成corner
def check_corners(bonds):
    keys = list(bonds.keys())
    corner_variables = {}
    # print("Processing keys:", keys)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1, key2 = keys[i], keys[j]
            # 提取端点
            points1 = extract_points(key1)
            points2 = extract_points(key2)
            # print(f"\nComparing keys: {key1}, {key2}")
            # print(f"Endpoints for {key1}: {points1}")
            # print(f"Endpoints for {key2}: {points2}")
            # 找共有端点
            common_points = set(points1).intersection(points2)
            # print(f"Common points: {common_points}")
            if common_points:
                common_point = common_points.pop()  # 提取共有端点
                # 确定不同的端点
                point1 = (set(points1) - {common_point}).pop()
                point2 = (set(points2) - {common_point}).pop()
                # print(f"Unique points: {point1}, {point2} from common point: {common_point}")
                # 检查第一位是否不同
                if point1[0] != point2[0]:
                    corner_name = f"corner_{common_point}_to_{point1}_and_{point2}"
                    corner_var = Binary(corner_name)
                    corner_variables[corner_name] = (corner_var, bonds[key1], bonds[key2])
                    # print(f"Formed corner: {corner_name}")
             
    return corner_variables

# H_corner
def create_hamiltonian(corner_variables):
    H = 0
    for corner_name, (corner_var, bond_var1, bond_var2) in corner_variables.items():
        # 定义corner对应的哈密顿量约束
        H += 3 * corner_var + bond_var1 * bond_var2 - 2 * corner_var * (bond_var1 + bond_var2)
    return H
corner_variables = check_corners(bonds)
H_corner = create_hamiltonian(corner_variables)
# H_diag
# 添加对角约束
def add_diagonal_constraints(corner_variables):
    H_diag = 0
    corners = list(corner_variables.keys())
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            c1 = corners[i]
            c2 = corners[j]
            # 提取角点坐标和核心点
            parts1 = c1.split('_to_')
            core1 = parts1[0].split('_')[-1]
            ends1 = parts1[1].split('_and_')
            parts2 = c2.split('_to_')
            core2 = parts2[0].split('_')[-1]
            ends2 = parts2[1].split('_and_')
            p1, p2 = ends1
            p3, p4 = ends2
            # print(f"Comparing: {c1} with {c2}")
            # print(f"Coordinates: {p1}, {p2} vs {p3}, {p4}")
            # print(f"Cores: {core1} vs {core2}")
            # 检查核心点的数位差异
            if abs(int(core1[0]) - int(core2[0])) == 1 and abs(int(core1[1]) - int(core2[1])) == 1:
                if {p1, p2} == {p3, p4}:
                    H_diag += corner_variables[c1][0] * corner_variables[c2][0]
                    # print(f"Adding constraint between {c1} and {c2}")
    return H_diag
H_diag = add_diagonal_constraints(corner_variables)

# print("\nHamiltonian:", H_corner)
# 添加相邻约束
def add_adjacent(corner_variables):
    H_adjacent = 0
    corners = list(corner_variables.keys())
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            c1 = corners[i]
            c2 = corners[j]
            # 提取角点坐标和核心点
            parts1 = c1.split('_to_')
            core1 = parts1[0].split('_')[-1]
            ends1 = parts1[1].split('_and_')
            parts2 = c2.split('_to_')
            core2 = parts2[0].split('_')[-1]
            ends2 = parts2[1].split('_and_')
            p1, p2 = ends1
            p3, p4 = ends2
            # print(f"Comparing: {c1} with {c2}")
            # print(f"Coordinates: {p1}, {p2} vs {p3}, {p4}")
            # print(f"Cores: {core1} vs {core2}")
            # 检查核心点是否相同
            if core1 == core2:
                # 检查坐标不完全相同
                if {p1, p2} != {p3, p4}:
                    H_adjacent += corner_variables[c1][0] * corner_variables[c2][0]
                    # print(f"Adding adjacent constraint between {c1} and {c2}")
    return H_adjacent
H_adjacent = add_adjacent(corner_variables)
# Apply the workaround for type compatibility: use negative and double negative
H = A_mono*H_node + A_bond*H_bond_all + A_corner*H_corner - ((-1)*A_bond*H_bond) +A_diag*H_diag +A_adjacent * H_adjacent
# print(H.compile().to_qubo()) # doctest: +SKIP


# Compile the model
start_time = time.time()
model = H.compile()
compilation_time = time.time() - start_time
bqm = model.to_bqm()

# Record memory usage after compilation
memory_usage_after_compilation = psutil.Process().memory_info().rss / 1024 ** 2  # in MB

# Solve using simulated annealing
sa = neal.SimulatedAnnealingSampler()
num_reads = 10  # Ensure num_reads is defined as an integer
start_time = time.time()
sampleset = sa.sample(bqm, num_reads=num_reads)
sampling_time = time.time() - start_time

# Record memory usage after sampling
memory_usage_after_sampling = psutil.Process().memory_info().rss / 1024 ** 2  # in MB

decoded_samples = model.decode_sampleset(sampleset)

# Plot each of the 10 solutions and save the plots
plot_start_time = time.time()
def plot_lattice_with_bonds(configuration_dict, filename):
    # Extract node configuration
    node_configuration = [configuration_dict[f'x[{i}][{j}]'] for i in range(num_rows) for j in range(num_cols)]
    bond_configuration = [configuration_dict[key] for key in bonds.keys()]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-0.5, num_cols - 0.5)
    ax.set_ylim(-0.5, num_rows - 0.5)
    ax.set_aspect('equal')

    # Lattice positions for nodes
    lattice_coords = [(col, 1 - row) for row in range(num_rows) for col in range(num_cols)]
    
    # Bond connections between nodes
    bond_connections = []
    for row in range(num_rows):
        for col in range(num_cols - 1):
            bond_connections.append((row * num_cols + col, row * num_cols + col + 1))
    for col in range(num_cols):
        bond_connections.append((col, num_cols + col))

    # Plot bonds
    for i, (start, end) in enumerate(bond_connections):
        if bond_configuration[i] == 1:
            x_coords = [lattice_coords[start][0], lattice_coords[end][0]]
            y_coords = [lattice_coords[start][1], lattice_coords[end][1]]
            ax.plot(x_coords, y_coords, color='blue', linewidth=2)

    # Plot lattice points
    for i, coord in enumerate(lattice_coords):
        color = 'red' if node_configuration[i] == 1 else 'gray'
        ax.plot(*coord, 'o', markersize=20, color=color)
        ax.text(coord[0], coord[1] + 0.2, f'x{i+1}', ha='center', fontsize=12)

    # Set title
    plt.grid(True)
    plt.savefig(filename)
    plt.close(fig)


# Loop over the 10 samples and save the plots
for idx, sample in enumerate(decoded_samples):
    filename = os.path.join(output_dir, f'sample_{idx + 1}.png')
    plot_lattice_with_bonds(sample.sample, filename)
plot_time = time.time() - plot_start_time

# Record memory usage after plotting
memory_usage_after_plotting = psutil.Process().memory_info().rss / 1024 ** 2  # in MB

# print(corner_variables)

# Write information to the info file
with open(info_file, "w") as f:
    f.write(f"Compilation time: {compilation_time:.4f} seconds\n")
    f.write(f"Sampling time: {sampling_time:.4f} seconds\n")
    f.write(f"Plotting time: {plot_time:.4f} seconds\n")
    f.write(f"Total execution time: {compilation_time + sampling_time + plot_time:.4f} seconds\n")
    f.write(f"Number of reads: {num_reads}\n")
    f.write(f"Memory usage after compilation: {memory_usage_after_compilation:.2f} MB\n")
    f.write(f"Memory usage after sampling: {memory_usage_after_sampling:.2f} MB\n")
    f.write(f"Memory usage after plotting: {memory_usage_after_plotting:.2f} MB\n")
    f.write(f"A_mono: {A_mono}\n")
    f.write(f"A_bond: {A_bond}\n")
    f.write("Corner names:\n")
    f.write("\n".join(corner_variables) + "\n")  # 每个角的名称占一行
    f.write(f"Point chains: {list(bonds.keys())}\n")  # Record the point chains
    f.write(f"QUBO representation:\n{H.compile().to_qubo()}\n")  # 将 QUBO 表示写入文件
    f.write(f"H_corner: {H_corner}\n")  # 将 H_corner 的信息写入文件

    # Record each sample's binary string and variable names
    f.write("\nSample Results:\n")
    for idx, sample in enumerate(decoded_samples):
        f.write(f"\nSample {idx + 1}:\n")
        
        # Create a string of the binary values for each variable
        binary_string = ''.join(str(int(value)) for value in sample.sample.values())
        variable_names = ', '.join(sample.sample.keys())
        
        # Write binary string and corresponding variable names to the file
        f.write(f"Binary String: {binary_string}\n")
        f.write(f"Variables: {variable_names}\n")


