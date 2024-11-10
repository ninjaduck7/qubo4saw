import re
from pyqubo import Binary, Constraint

# 定义连接键
bonds = {
    "x_00_to_01": Binary("x_00_to_01"), "x_01_to_02": Binary("x_01_to_02"),
    "x_10_to_11": Binary("x_10_to_11"), "x_11_to_12": Binary("x_11_to_12"),
    "x_00_to_10": Binary("x_00_to_10"), "x_01_to_11": Binary("x_01_to_11"), "x_02_to_12": Binary("x_02_to_12")
}

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
                if point1[0] != point2[0]:
                    corner_name = f"corner_{common_point}_to_{point1}_and_{point2}"
                    corner_var = Binary(corner_name)
                    corner_variables[corner_name] = (corner_var, bonds[key1], bonds[key2])
    return corner_variables

# 创建哈密顿量
def create_hamiltonian(corner_variables):
    H = 0
    for corner_name, (corner_var, bond_var1, bond_var2) in corner_variables.items():
        H += 3 * corner_var + bond_var1 * bond_var2 - 2 * corner_var * (bond_var1 + bond_var2)
    return H

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
            print(f"Comparing: {c1} with {c2}")
            print(f"Coordinates: {p1}, {p2} vs {p3}, {p4}")
            print(f"Cores: {core1} vs {core2}")
            # 检查核心点的数位差异
            if abs(int(core1[0]) - int(core2[0])) == 1 and abs(int(core1[1]) - int(core2[1])) == 1:
                if {p1, p2} == {p3, p4}:
                    H_diag += corner_variables[c1][0] * corner_variables[c2][0]
                    print(f"Adding constraint between {c1} and {c2}")
    return H_diag

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
            print(f"Comparing: {c1} with {c2}")
            print(f"Coordinates: {p1}, {p2} vs {p3}, {p4}")
            print(f"Cores: {core1} vs {core2}")
            # 检查核心点是否相同
            if core1 == core2:
                # 检查坐标不完全相同
                if {p1, p2} != {p3, p4}:
                    H_adjacent += corner_variables[c1][0] * corner_variables[c2][0]
                    print(f"Adding adjacent constraint between {c1} and {c2}")
    return H_adjacent


# 输出所有可能的corners及其哈密顿量
corner_variables = check_corners(bonds)
H_corner = create_hamiltonian(corner_variables)
H_diag = add_diagonal_constraints(corner_variables)
H_adjacent = add_adjacent(corner_variables)
# print("\nHamiltonian:", H_corner)
# print("Diagonal Constraints Hamiltonian:", H_diag)
print("Adjacent Constraints Hamiltonian:", H_adjacent)