import os
import numpy as np
import matplotlib.pyplot as plt
from pyqubo import Binary, Constraint
import neal
from itertools import combinations


# 设置参数
fig_width = 21

# 设置网格参数
N = 60
ratio_y = np.sqrt(3) / 2  # cos(60°)
N_Y = int(np.sqrt(N) / ratio_y)
N_X = N // N_Y

# 生成三角形网格的点
xv, yv = np.meshgrid(np.arange(N_X), np.arange(N_Y), sparse=False, indexing='xy')
yv = yv * ratio_y
xv = xv.astype(float)  # 将xv数组转换为float类型
xv[::2, :] += 1 / 2    # 平移操作

N = 50 # 可以调整N的值
L = N - 1  # 设置目标激活边数量，可以调整

# 定义每个约束的系数
c_nodes = 1.0
c_farthest_points = 2.0
c_bonds = 1.5
c_edge_activation = 5.0
c_farthest_point_edges = 1.0
c_other_points_edges = 1.0
c_triangular = 5.0


# 设置采样次数
num_samples = 20  # 可以调整采样次数

# 编码节点为PyQUBO变量
nodes = {}
for j in range(N_Y):
    for i in range(N_X):
        nodes[(i, j)] = Binary(f"x_{i}{j}")


edges = {}
# 为每条边定义PyQUBO二进制变量
for j in range(N_Y):
    for i in range(N_X):
        # 右边的边
        if i < N_X - 1:
            edges[((i, j), (i + 1, j))] = Binary(f"edge_{i}{j}_{i+1}{j}")
        # 上边
        if j < N_Y - 1:
            edges[((i, j), (i, j + 1))] = Binary(f"edge_{i}{j}_{i}{j+1}")
        # 右上方的边（适用于偶数行）
        if i < N_X - 1 and j < N_Y - 1 and j % 2 == 0:
            edges[((i, j), (i + 1, j + 1))] = Binary(f"edge_{i}{j}_{i+1}{j+1}")
        # 左上方的边（适用于奇数行）
        if i > 0 and j < N_Y - 1 and j % 2 == 1:
            edges[((i, j), (i - 1, j + 1))] = Binary(f"edge_{i}{j}_{i-1}{j+1}")

# 统计总共多少点
total_points = len(nodes)
print(f"总共的点数: {total_points}")

# 统计总共多少边
total_edges = len(edges)
print(f"总共的边数: {total_edges}")


# 计算所有节点之间的距离
distances = {}
for (i1, j1), (i2, j2) in combinations(nodes.keys(), 2):
    dist = np.sqrt((xv[j1, i1] - xv[j2, i2]) ** 2 + (yv[j1, i1] - yv[j2, i2]) ** 2)
    distances[((i1, j1), (i2, j2))] = dist

# 找出距离最远的两个点
(farthest_point1, farthest_point2) = max(distances, key=distances.get)

# 记录最远的两个点
print(f"最远的两个点: {farthest_point1}, {farthest_point2}")

# 构建 H_nodes 时排除最远的两个点
sum_nodes = sum(nodes[(i, j)] for i in range(N_X) for j in range(N_Y) 
                if (i, j) != farthest_point1 and (i, j) != farthest_point2)

# 设置哈密顿量 H = (sum(nodes) - N)^2
H_nodes = Constraint((sum_nodes - (N - 2)) ** 2, label="H_nodes")


# 为最远的两个点添加额外约束，强制其值为1
H_farthest_points = Constraint((1 - nodes[farthest_point1]) ** 2 + (1 - nodes[farthest_point2]) ** 2, label="H_farthest_points")
# 构建边的哈密顿量约束 H_bonds
sum_edges = sum(edges[edge] for edge in edges)
H_bonds = Constraint((sum_edges - L) ** 2, label="H_bonds")


# 添加新的约束：要求激活边的两端的点必须都激活
H_edge_activation = sum(
    edges[edge] * (1 - nodes[edge[0]]) + edges[edge] * (1 - nodes[edge[1]])
    for edge in edges
)

# 从之前定义的 edges 中找到最远的两个点的周边边
farthest_edges = []

# 从之前定义的 edges 中找到每个最远点的相邻边
farthest_point1_edges = []
farthest_point2_edges = []

# 遍历 edges，找到与每个最远点相邻的边
for edge in edges:
    # 检查是否包含最远的第一个点
    if edge[0] == farthest_point1 or edge[1] == farthest_point1:
        farthest_point1_edges.append(edges[edge])
    # 检查是否包含最远的第二个点
    if edge[0] == farthest_point2 or edge[1] == farthest_point2:
        farthest_point2_edges.append(edges[edge])

# 构建每个最远点的约束，使得相邻边中只能有一条激活
sum_farthest_point1_edges = sum(farthest_point1_edges)
H_farthest_point1_edges = Constraint((sum_farthest_point1_edges - 1) ** 2, label="H_farthest_point1_edges")

sum_farthest_point2_edges = sum(farthest_point2_edges)
H_farthest_point2_edges = Constraint((sum_farthest_point2_edges - 1) ** 2, label="H_farthest_point2_edges")

# 遍历所有点，排除最远的两个点
H_other_points_edges = 0  # 初始化用于存储所有非最远点的边约束

for point in nodes:
    # 跳过最远的两个点
    if point == farthest_point1 or point == farthest_point2:
        continue
    
    # 获取当前点的相邻边
    neighbor_edges = []
    for edge in edges:
        if edge[0] == point or edge[1] == point:
            neighbor_edges.append(edges[edge])
    
    # 对于当前点构建约束 (邻边和 - 2)^2
    sum_neighbor_edges = sum(neighbor_edges)
    H_other_points_edges += (sum_neighbor_edges - 2) ** 2

# 将上述约束打包成 Constraint
H_other_points_edges = Constraint(H_other_points_edges, label="H_other_points_edges")


# 添加用于三角形闭包的辅助变量
auxiliary = {}

# 创建一个存储三角形约束的总和变量
H_triangles = 0

# 遍历所有偶数行和奇数行的小闭合三角形
for j in range(N_Y):
    for i in range(N_X):
        if j % 2 == 0:
            # 对于偶数行上的三角形，例如 (0,0)-(1,0), (0,0)-(1,1), (1,0)-(1,1)
            if i < N_X  and j < N_Y :
                edge1 = edges.get(((i, j), (i + 1, j)))
                edge2 = edges.get(((i, j), (i + 1, j + 1)))
                edge3 = edges.get(((i + 1, j), (i + 1, j + 1)))

                if edge1 and edge2 and edge3:
                    # 引入辅助量子比特 s 用于三阶张量项
                    s = Binary(f"s_{i}_{j}_{i+1}_{j}_{i+1}_{j+1}")
                    H_triangles +=  edge3 * s + 2 * (3 * s + edge1 * edge2 - 2*edge1*s -2*edge2*s)
        else:
            # 对于奇数行上的三角形，例如 (0,1)-(1,1), (0,0)-(0,1), (0,0)-(1,1)
            if i < N_X  and j > 0:
                edge1 = edges.get(((i, j), (i + 1, j)))
                edge2 = edges.get(((i, j), (i, j - 1)))
                edge3 = edges.get(((i, j - 1), (i + 1, j)))

                if edge1 and edge2 and edge3:
                    # 引入辅助量子比特 s 用于三阶张量项
                    s = Binary(f"s_{i}_{j}_{i+1}_{j}_{i}_{j-1}")
                    H_triangles +=  edge3 * s + 2 * (3 * s + edge1 * edge2 - 2*edge1*s -2*edge2*s)
# print(H_triangles)


# 添加特定的三角形配置的哈密顿量项
H_special_triangles = 0
# 定义特定三角形约束的系数
c_special_triangles = c_triangular


# 遍历所有行来添加特定三角形配置的情况
for j in range(N_Y):
    # 奇数行的第一个边的三角形
    if j % 2 == 1 and j < N_Y :
        i = 0
        edge1 = edges.get(((i, j), (i + 1, j)))
        edge2 = edges.get(((i, j), (i, j + 1)))
        edge3 = edges.get(((i + 1, j), (i, j + 1)))

        if edge1 and edge2 and edge3:
            s = Binary(f"s_triangle_up_first_{i}_{j}")
            H_special_triangles += edge3 * s + 2 * (3 * s + edge1 * edge2 - 2*edge1*s -2*edge2*s)

    # 偶数行的最后一个边的三角形
    if j % 2 == 0 and j > 0:
        i = N_X - 1 # 假设至少有2列
        edge1 = edges.get(((i, j), (i + 1, j)))
        edge2 = edges.get(((i + 1, j), (i + 1, j - 1)))
        edge3 = edges.get(((i, j), (i + 1, j - 1)))

        if edge1 and edge2 and edge3:
            s = Binary(f"s_triangle_down_last_{i}_{j}")
            H_special_triangles += edge3 * s + 2 * (3 * s + edge1 * edge2 - 2*edge1*s -2*edge2*s)
# 在定义最终哈密顿量 H 时，直接对每个约束乘上相应系数
H = (c_nodes * H_nodes +
     c_bonds * H_bonds +
     c_edge_activation * H_edge_activation +
     c_farthest_points * H_farthest_points +
     c_farthest_point_edges * H_farthest_point1_edges +
     c_farthest_point_edges * H_farthest_point2_edges +
     c_other_points_edges * H_other_points_edges)

# 将三角形约束加入总哈密顿量
H += c_triangular * H_triangles
H += c_special_triangles * H_special_triangles
# 编译模型
model = H.compile()
qubo, offset = model.to_qubo()

# 创建存储图片和输出结果的文件夹
output_dir = "infor"
os.makedirs(output_dir, exist_ok=True)

# 打开文本文件以记录所有采样的激活状态
output_txt_path = os.path.join(output_dir, "triangular_grid_points_edges_activation_all_samples.txt")
with open(output_txt_path, 'w') as f:
    f.write(f"Triangular constraint: {H_triangles}\n") 
    # 运行多次采样
    sampler = neal.SimulatedAnnealingSampler()
    for sample_num in range(num_samples):
        response = sampler.sample_qubo(qubo, num_reads=1, beta_range=(0.1, 5.0), num_sweeps=1000)
        solution = response.first.sample
    
        # 写入当前采样的标号
        f.write(f"Sampling {sample_num + 1} Activation Status:\n")

        # 记录每个点的激活状态
        f.write("Points Activation:\n")
        for j in range(N_Y):
            for i in range(N_X):
                # 获取采样结果，检查节点是否被激活
                is_active = solution.get(f"x_{i}{j}", 0) == 1
                status = "active" if is_active else "inactive"
                # 写入点的状态到文本文件
                f.write(f"Point ({i},{j}): Status: {status}\n")

        # 记录每条边的激活状态
        f.write("Edge Activation:\n")
        for edge, var_name in edges.items():
            (i1, j1), (i2, j2) = edge
            is_active = solution.get(f"edge_{i1}{j1}_{i2}{j2}", 0) == 1
            status = "active" if is_active else "inactive"
            # 写入边的状态到文本文件
            f.write(f"Edge ({i1},{j1})-({i2},{j2}): Status: {status}\n")
        f.write("\n")  # 分隔不同采样的内容

        # 绘制当前采样的三角形网格并保存图片
        fig, ax = plt.subplots(figsize=(fig_width, fig_width))
        ax.scatter(xv, yv)

        # 绘制所有点的激活状态
        for j in range(N_Y):
            for i in range(N_X):
                # 标记点的编号
                ax.text(xv[j, i], yv[j, i], f'({i},{j})', fontsize=8, ha='center', va='center', color='red')
                
                # 如果该点被激活，用绿色圈出
                if solution.get(f"x_{i}{j}", 0) == 1:
                    ax.scatter(xv[j, i], yv[j, i], s=700, facecolors='none', edgecolors='green', linewidth=2)

        # 绘制每条边的激活状态
        for edge, var_name in edges.items():
            (i1, j1), (i2, j2) = edge
            is_active = solution.get(f"edge_{i1}{j1}_{i2}{j2}", 0) == 1
            color = 'red' if is_active else 'black'
            linewidth = 2 if is_active else 0.5
            ax.plot([xv[j1, i1], xv[j2, i2]], [yv[j1, i1], yv[j2, i2]], color=color, linewidth=linewidth)

        # 保存当前采样的图片
        output_path_with_labels = os.path.join(output_dir, f"triangular_grid_with_labels_sample_{sample_num + 1}.png")
        plt.savefig(output_path_with_labels)
        plt.close()

print(f"所有采样结果已保存至 {output_dir} 文件夹。")