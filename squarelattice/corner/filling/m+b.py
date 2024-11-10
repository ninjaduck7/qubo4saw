import re
from pprint import pprint
from pyqubo import Array, Binary
import neal
import os
# 设置格子尺寸
num_rows = num_cols = 6
filling_rate = 0.7  # 设置填充率

# 计算 N 和 L
N = int(filling_rate * num_rows * num_cols)
L = N - 1


# Delete the information file if it exists and create a new one
activation_file = "nodes+bonds.txt"
if os.path.exists(activation_file):
    os.remove(activation_file)



# Define binary variables for nodes and bonds using Array to make scaling easier
nodes = Array.create('x', shape=(num_rows, num_cols), vartype='BINARY')
# print(nodes)
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

# Define the Hamiltonian for the nodes and bonds
A_mono = 1
A_bond_all = 1
A_corner = 1
A_diag = 1
A_bond = 1
A_adjacent = 2
num_reads = 20

# Define node constraint Hamiltonian
H_node = A_mono * (sum([nodes[row, col] for row in range(num_rows) for col in range(num_cols)]) - N) ** 2
H_bond_all = A_bond * (sum(bonds.values()) - L) ** 2
H_bond = 0

for row in range(num_rows):
    for col in range(num_cols - 1):
        bond_var = bonds[f'x_{row}{col}_to_{row}{col+1}']
        H_bond += bond_var * (1 - nodes[row, col]) + bond_var * (1 - nodes[row, col + 1])

for row in range(num_rows - 1):
    for col in range(num_cols):
        bond_var = bonds[f'x_{row}{col}_to_{row+1}{col}']
        H_bond += bond_var * (1 - nodes[row, col]) + bond_var * (1 - nodes[row + 1, col])

# Regular expressions and corner checks
def extract_points(key):
    return tuple(re.findall(r'\d{2}', key))

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

def create_hamiltonian(corner_variables):
    H = 0
    for corner_name, (corner_var, bond_var1, bond_var2) in corner_variables.items():
        H += 3 * corner_var + bond_var1 * bond_var2 - 2 * corner_var * (bond_var1 + bond_var2)
    return H

corner_variables = check_corners(bonds)
H_corner = create_hamiltonian(corner_variables)

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

H = A_mono * H_node + A_bond_all * H_bond_all + A_corner * H_corner - ((-1) * A_bond * H_bond) + A_diag * H_diag + A_adjacent * H_adjacent
model = H.compile()
bqm = model.to_bqm()

# Solve using simulated annealing
sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=num_reads, sweeps=1000, beta_range=(5, 50))
decoded_samples = model.decode_sampleset(sampleset)
print(decoded_samples)
# Write the activation information to the file in the required format
with open(activation_file, "w") as f:
    for idx, sample in enumerate(decoded_samples):
        f.write(f"Sample {idx + 1}:\n")
        # Extract and write activated nodes and bonds
        for variable, value in sample.sample.items():
            if value == 1:  # Only record activated variables
                if re.match(r"x\[\d+\]\[\d+\]", variable):  # Node format x[5][1]
                    f.write(f"{variable}:1\n")
                elif re.match(r"x_\d+_to_\d+", variable):  # Bond format x_45_to_55
                    f.write(f"{variable}:1\n")
        f.write("\n")

print("Activation information has been saved in 'nodes+bonds.txt'.")