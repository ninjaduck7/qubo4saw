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
info_file = "3x3infor.txt"
if os.path.exists(info_file):
    os.remove(info_file)

# Define the lattice size for 3x3 grid
num_rows = 3
num_cols = 3

# Define binary variables for nodes and bonds using Array to make scaling easier
nodes = Array.create('x', shape=(num_rows, num_cols), vartype='BINARY')
bonds = {}

# Define bond variables (horizontal, vertical)
# Horizontal bonds (between columns in the same row)
for row in range(num_rows):
    for col in range(num_cols - 1):
        bonds[f'x_{row}{col}_to_{row}{col+1}'] = Binary(f'x_{row}{col}_to_{row}{col+1}')

# Vertical bonds (between rows in the same column)
for row in range(num_rows - 1):
    for col in range(num_cols):
        bonds[f'x_{row}{col}_to_{row+1}{col}'] = Binary(f'x_{row}{col}_to_{row+1}{col}')

# Define the Hamiltonian for the nodes
N = 9  # Set the total number of active nodes (updated for 3x3)
L = 8  # Set the total number of active bonds (updated for 3x3)
A_mono = 1
A_bond_all = 10
A_corner = 1
A_diag = 1
A_bond = 1
A_adjacent = 1

# Define node constraint Hamiltonian
H_node = A_mono * (sum([nodes[row, col] for row in range(num_rows) for col in range(num_cols)]) - N) ** 2

# Define bond constraint Hamiltonian
H_bond_all = A_bond * (sum(bonds.values()) - L) ** 2

# Define each bond Hamiltonian term separately to ensure compatibility
H_bond = 0

# Add bonds within the same layer (horizontal and vertical)
for row in range(num_rows):
    for col in range(num_cols - 1):
        bond_var = bonds[f'x_{row}{col}_to_{row}{col+1}']
        H_bond += bond_var * (1 - nodes[row, col]) + bond_var * (1 - nodes[row, col + 1])

for row in range(num_rows - 1):
    for col in range(num_cols):
        bond_var = bonds[f'x_{row}{col}_to_{row+1}{col}']
        H_bond += bond_var * (1 - nodes[row, col]) + bond_var * (1 - nodes[row + 1, col])

# 正则表达式来提取坐标
def extract_points(key):
    return tuple(re.findall(r'\d{2}', key))

# 检查是否可以形成corner
def check_corners(bonds):
    keys = list(bonds.keys())
    corner_variables = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1, key2 = keys[i], keys[j]
            points1 = extract_points(key1)
            points2 = extract_points(key2)
            common_points = set(points1).intersection(points2)
            if common_points:
                common_point = common_points.pop()
                point1 = (set(points1) - {common_point}).pop()
                point2 = (set(points2) - {common_point}).pop()
                if point1[0] != point2[0] and point1[1] != point2[1]:

                    corner_name = f"corner_{common_point}_to_{point1}_and_{point2}"
                    corner_var = Binary(corner_name)
                    corner_variables[corner_name] = (corner_var, bonds[key1], bonds[key2])
    return corner_variables

# H_corner
def create_hamiltonian(corner_variables):
    H = 0
    for corner_name, (corner_var, bond_var1, bond_var2) in corner_variables.items():
        H += 3 * corner_var + bond_var1 * bond_var2 - 2 * corner_var * (bond_var1 + bond_var2)
    return H

corner_variables = check_corners(bonds)
H_corner = create_hamiltonian(corner_variables)

# H_diag: 添加对角约束
def add_diagonal_constraints(corner_variables):
    H_diag = 0
    corners = list(corner_variables.keys())
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            c1 = corners[i]
            c2 = corners[j]
            parts1 = c1.split('_to_')
            core1 = parts1[0].split('_')[-1]
            ends1 = parts1[1].split('_and_')
            parts2 = c2.split('_to_')
            core2 = parts2[0].split('_')[-1]
            ends2 = parts2[1].split('_and_')
            p1, p2 = ends1
            p3, p4 = ends2
            if abs(int(core1[0]) - int(core2[0])) == 1 and abs(int(core1[1]) - int(core2[1])) == 1:
                if {p1, p2} == {p3, p4}:
                    H_diag += corner_variables[c1][0] * corner_variables[c2][0]
    return H_diag

H_diag = add_diagonal_constraints(corner_variables)

# H_adjacent: 添加相邻约束
def add_adjacent(corner_variables):
    H_adjacent = 0
    corners = list(corner_variables.keys())
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            c1 = corners[i]
            c2 = corners[j]
            parts1 = c1.split('_to_')
            core1 = parts1[0].split('_')[-1]
            ends1 = parts1[1].split('_and_')
            parts2 = c2.split('_to_')
            core2 = parts2[0].split('_')[-1]
            ends2 = parts2[1].split('_and_')
            p1, p2 = ends1
            p3, p4 = ends2
            if core1 == core2:
                if {p1, p2} != {p3, p4}:
                    H_adjacent += corner_variables[c1][0] * corner_variables[c2][0]
    return H_adjacent

H_adjacent = add_adjacent(corner_variables)

# Apply the workaround for type compatibility: use negative and double negative
H = A_mono * H_node + A_bond * H_bond_all + A_corner * H_corner - ((-1) * A_bond * H_bond) + A_diag * H_diag + A_adjacent * H_adjacent

# Compile the model
start_time = time.time()
model = H.compile()
compilation_time = time.time() - start_time
bqm = model.to_bqm()

# Record memory usage after compilation
memory_usage_after_compilation = psutil.Process().memory_info().rss / 1024 ** 2  # in MB

# Solve using simulated annealing
sa = neal.SimulatedAnnealingSampler()
num_reads = 10
start_time = time.time()
sampleset = sa.sample(bqm, num_reads=num_reads)
sampling_time = time.time() - start_time

# Record memory usage after sampling
memory_usage_after_sampling = psutil.Process().memory_info().rss / 1024 ** 2  # in MB

decoded_samples = model.decode_sampleset(sampleset)



# Write the information to the info file
with open(info_file, "w") as f:
    f.write(f"Compilation time: {compilation_time:.4f} seconds\n")
    f.write(f"Sampling time: {sampling_time:.4f} seconds\n")
    f.write(f"Number of reads: {num_reads}\n")
    f.write(f"Memory usage after compilation: {memory_usage_after_compilation:.2f} MB\n")
    f.write(f"Memory usage after sampling: {memory_usage_after_sampling:.2f} MB\n")
    f.write(f"A_mono: {A_mono}\n")
    f.write(f"A_bond: {A_bond}\n")
    f.write("Corner names:\n")
    f.write("\n".join(corner_variables) + "\n")
    f.write(f"Point chains: {list(bonds.keys())}\n")
    f.write(f"QUBO representation:\n{H.compile().to_qubo()}\n")
    f.write(f"H_corner: {H_corner}\n")
    f.write(f"H_adjacent: {H_adjacent}\n")
    f.write(f"H_bond: {H_bond}\n")
    f.write(f"H_bond_all: {H_bond_all}\n")
    f.write(f"H_diag: {H_diag}\n")
    f.write("\nSample Results:\n")
    for idx, sample in enumerate(decoded_samples):
        f.write(f"\nSample {idx + 1}:\n")
        binary_string = ''.join(str(int(value)) for value in sample.sample.values())
        variable_names = ', '.join(sample.sample.keys())
        f.write(f"Binary String: {binary_string}\n")
        f.write(f"Variables: {variable_names}\n")


import matplotlib.pyplot as plt
import re
import os

# Define the lattice size for grid
def draw_lattice(sample, idx, num_rows, num_cols):
    # Create a new figure
    plt.figure(figsize=(6, 6))
    plt.title(f"Sample {idx + 1} - Lattice Activation")
    
    # Draw the grid points
    for i in range(num_rows):
        for j in range(num_cols):
            plt.plot(j, -i, 'ko', markersize=10)  # plot points as black circles
    
    # Draw activated bonds (edges)
    for variable, value in sample.sample.items():
        if value == 1:  # Only draw if the binary value is 1
            if "to" in variable:  # This is a bond variable
                points = re.findall(r'x_(\d+)(\d+)_to_(\d+)(\d+)', variable)
                if points:  # Ensure there are enough values to unpack
                    row1, col1, row2, col2 = map(int, points[0])
                    plt.plot([col1, col2], [-row1, -row2], 'c-', linewidth=3)  # Draw bond as a cyan line for better visibility
                    # Mark the activated bond
                    plt.text((col1 + col2) / 2, (-row1 + -row2) / 2, '1', color='cyan', fontsize=12, ha='center', weight='bold')
            elif "x[" in variable:  # This is a node variable
                points = re.findall(r'\d+', variable)
                if len(points) == 2:  # Ensure there are enough values to unpack
                    row, col = map(int, points)
                    plt.plot(col, -row, 'ro', markersize=12)  # Highlight activated nodes as red circles

    # Set axis limits and labels
    plt.xlim(-0.5, num_cols - 0.5)
    plt.ylim(-num_rows + 0.5, 0.5)
    plt.xticks(range(num_cols))
    plt.yticks([-i for i in range(num_rows)])
    plt.gca().set_aspect('equal')  # Equal aspect ratio to ensure square grid
    plt.grid(True)
    
    # Save the figure to the output directory
    plt.savefig(os.path.join(output_dir, f"lattice_sample_{idx + 1}.png"))
    plt.close()

# Iterate through the decoded samples and draw each one
for idx, sample in enumerate(decoded_samples):
    draw_lattice(sample, idx, num_rows, num_cols)

print("Lattice drawings have been saved in the 'pic' directory.")
