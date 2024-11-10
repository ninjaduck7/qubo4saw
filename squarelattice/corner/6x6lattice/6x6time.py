import re
from pprint import pprint
from pyqubo import Array, Binary
import neal
import matplotlib.pyplot as plt
import os
import shutil
import time
import psutil

# # Delete the output directory if it exists and create it again
# output_dir = "pic"
# if os.path.exists(output_dir):
#     shutil.rmtree(output_dir)
# os.makedirs(output_dir, exist_ok=True)


# Define the lattice size for 3x3 grid
num_rows = 6
num_cols = 6

# Delete the information file if it exists and create a new one
info_file = f"{num_rows}x{num_cols}infor.txt"
if os.path.exists(info_file):
    os.remove(info_file)



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
# A_bond_all for summation of all bonds
# A_bond for constraints on monomers activation and bond activation

N = 21  # Set the total number of active nodes (updated for 3x3)
L = 20# Set the total number of active bonds (updated for 3x3)

# for total monomers number
A_mono = 1
# for total bonds number
A_bond_all = 1

# for bonds and corners constraint
A_corner = 1

# for 4-loops exclusion
A_diag = 1

# for monomer and bonds constraint
A_bond = 1

# for self-avoidance
A_adjacent = 2
# for samoling times
num_reads = 20

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
H = A_mono * H_node + A_bond_all * H_bond_all  + A_corner * H_corner - ((-1) * A_bond * H_bond) + A_diag * H_diag + A_adjacent * H_adjacent
# H_0 = A_mono * H_node + A_bond_all * H_bond_all - ((-1) * A_bond * H_bond)
# Compile the model
start_time = time.time()
model = H.compile(strength =50)
compilation_time = time.time() - start_time
bqm = model.to_bqm()

# Record memory usage after compilation
# memory_usage_after_compilation = psutil.Process().memory_info().rss / 1024 ** 2  # in MB

# Solve using simulated annealing
sa = neal.SimulatedAnnealingSampler()
sampling_times = []

num_runs = 20
for _ in range(num_runs):
    start_time = time.time()
    sampleset = sa.sample(bqm, num_reads=1, sweeps=1000, beta_range=(5, 50))
    elapsed_time = time.time() - start_time
    sampling_times.append(elapsed_time)

# Calculate average sampling time
average_time = sum(sampling_times) / len(sampling_times)

# Save the times to a file
time_file_path = "time.txt"
with open(time_file_path, 'w') as f:
    f.write("Sampling Times:\n")
    f.writelines(f"{time:.4f}s\n" for time in sampling_times)
    f.write(f"Average Sampling Time: {average_time:.4f}s\n")


