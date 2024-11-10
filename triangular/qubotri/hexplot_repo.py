# %%

import numpy as np

import matplotlib.pyplot as plt

  

# 设置参数

fig_width = 21

  

# 设置网格参数

N = 121

ratio_y = np.sqrt(3) / 2  # cos(60°)

N_Y = int(np.sqrt(N) / ratio_y)

N_X = N // N_Y

  

# 生成六边形网格的点

xv, yv = np.meshgrid(np.arange(N_X), np.arange(N_Y), sparse=False, indexing='xy')

yv = yv * ratio_y

xv = xv.astype(float)  # 将xv数组转换为float类型

xv[::2, :] += 0.5    # 平移奇数行

  

# 找到中心的点

center_idx_x = N_X // 2

center_idx_y = N_Y // 2

center_x = xv[center_idx_y, center_idx_x]

center_y = yv[center_idx_y, center_idx_x]

  

# 找到中心点的最近邻点（六个方向）

neighbors = [

    (center_x + 1, center_y),  # 右

    (center_x + 0.5, center_y + ratio_y),  # 右上

    (center_x - 0.5, center_y + ratio_y),  # 左上

    (center_x - 1, center_y),  # 左

    (center_x - 0.5, center_y - ratio_y),   # 左下

    (center_x + 0.5, center_y - ratio_y),  # 右下

]

  

# 从输入的二进制串生成激活的边

binary_string = "111111111111111111111111111111"  # 示例二进制串（长度为30）

assert len(binary_string) == 30, "二进制串长度必须为30"

  

# 定义边的名称和激活状态

edges = {

    # 内层六边形的六条边

    "x_inner_12": int(binary_string[0]),

    "x_inner_23": int(binary_string[1]),

    "x_inner_34": int(binary_string[2]),

    "x_inner_45": int(binary_string[3]),

    "x_inner_56": int(binary_string[4]),

    "x_inner_61": int(binary_string[5]),

  

    # 向外延伸的六条边（tunnel）

    "x_tunnel_1": int(binary_string[6]),

    "x_tunnel_2": int(binary_string[7]),

    "x_tunnel_3": int(binary_string[8]),

    "x_tunnel_4": int(binary_string[9]),

    "x_tunnel_5": int(binary_string[10]),

    "x_tunnel_6": int(binary_string[11]),

  

    # 外层与tunnel相交的12条边

    "x_1_outer_1": int(binary_string[12]),

    "x_1_outer_2": int(binary_string[13]),

    "x_2_outer_1": int(binary_string[14]),

    "x_2_outer_2": int(binary_string[15]),

    "x_3_outer_1": int(binary_string[16]),

    "x_3_outer_2": int(binary_string[17]),

    "x_4_outer_1": int(binary_string[18]),

    "x_4_outer_2": int(binary_string[19]),

    "x_5_outer_1": int(binary_string[20]),

    "x_5_outer_2": int(binary_string[21]),

    "x_6_outer_1": int(binary_string[22]),

    "x_6_outer_2": int(binary_string[23]),

  

    # 最外层连接的六条边（随机连接）

    "x_random_12": int(binary_string[24]),

    "x_random_23": int(binary_string[25]),

    "x_random_34": int(binary_string[26]),

    "x_random_45": int(binary_string[27]),

    "x_random_56": int(binary_string[28]),

    "x_random_61": int(binary_string[29]),

}

  

# 绘制网格并标注出中心及外围的六边形

fig, ax = plt.subplots(figsize=(fig_width, fig_width))

ax.scatter(xv, yv, color='lightblue', label='Grid Points')

  

# 绘制中心点

ax.scatter(center_x, center_y, color='red', s=100, label='Center Point')

  
  

# 绘制网格中的所有点之间的连接（黑色线段）

for j in range(N_Y):

    for i in range(N_X):

        # 连接右边的点

        if i < N_X - 1:

            ax.plot([xv[j, i], xv[j, i + 1]], [yv[j, i], yv[j, i + 1]], color='black', linestyle='-', linewidth=0.5)

        # 连接上面的点

        if j < N_Y - 1:

            ax.plot([xv[j, i], xv[j + 1, i]], [yv[j, i], yv[j + 1, i]], color='black', linestyle='-', linewidth=0.5)

        # 连接右上方的点（适用于奇数行）

        if i < N_X - 1 and j < N_Y - 1 and j % 2 == 0:

            ax.plot([xv[j, i], xv[j + 1, i + 1]], [yv[j, i], yv[j + 1, i + 1]], color='black', linestyle='-', linewidth=0.5)

          # 修正右上连接，避免把最上面最下面的点也连接起来

        if i < N_X - 1 and j > 0 and j % 2 == 0:

            ax.plot([xv[j, i], xv[j - 1, i + 1]], [yv[j, i], yv[j - 1, i + 1]], color='black', linestyle='-', linewidth=0.5)

  
  

# 连线形成中心的正六边形

hexagon = np.array(neighbors + [neighbors[0]])  # 闭合六边形

for i in range(6):

    if edges[f"x_inner_{i+1}{(i+2) if (i+2) <= 6 else 1}"]:

        ax.plot([neighbors[i][0], neighbors[(i + 1) % 6][0]],

                [neighbors[i][1], neighbors[(i + 1) % 6][1]],

                color='red', linestyle='-', linewidth=3, label=f'Inner Edge {i+1}-{(i+2) if (i+2) <= 6 else 1}')

  

# 绘制tunnel的六条边

outer_points = []

for i, neighbor in enumerate(neighbors):

    if edges[f"x_tunnel_{i+1}"]:

        extended_x = neighbor[0] + (neighbor[0] - center_x)

        extended_y = neighbor[1] + (neighbor[1] - center_y)

        ax.plot([neighbor[0], extended_x],

                [neighbor[1], extended_y],

                color='red', linestyle='-', linewidth=3, label=f'Tunnel Edge {i+1}')

        outer_points.append((extended_x, extended_y))

  

# 绘制外层相交的12条边

rotation_angle = np.pi / 3  # 60度旋转角度

cos_theta = np.cos(rotation_angle)

sin_theta = np.sin(rotation_angle)

outer_points_1 = []

outer_points_2 = []

for i, (extended_x, extended_y) in enumerate(outer_points):

    # 外层的中心点坐标

    outer_center_x = extended_x + (extended_x - neighbors[i][0])

    outer_center_y = extended_y + (extended_y - neighbors[i][1])

    # 逆时针旋转出外层端点x_outer_1

    rel_x, rel_y = extended_x - outer_center_x, extended_y - outer_center_y

    x_outer_1 = cos_theta * rel_x - sin_theta * rel_y + outer_center_x

    y_outer_1 = sin_theta * rel_x + cos_theta * rel_y + outer_center_y

    # 顺时针旋转出外层端点x_outer_2

    x_outer_2 = cos_theta * rel_x + sin_theta * rel_y + outer_center_x

    y_outer_2 = -sin_theta * rel_x + cos_theta * rel_y + outer_center_y

    outer_points_1.append((x_outer_1, y_outer_1))

    outer_points_2.append((x_outer_2, y_outer_2))

    if edges[f"x_{i+1}_outer_1"]:

        ax.plot([extended_x, x_outer_1],

                [extended_y, y_outer_1],

                color='red', linestyle='-', linewidth=3, label=f'Outer Edge {i+1}_1')

    if edges[f"x_{i+1}_outer_2"]:

        ax.plot([extended_x, x_outer_2],

                [extended_y, y_outer_2],

                color='red', linestyle='-', linewidth=3, label=f'Outer Edge {i+1}_2')

  

# 绘制随机连接皆六条边，将外侧端点连接

for i in range(6):

    if edges[f"x_random_{i+1}{(i+2) if (i+2) <= 6 else 1}"]:

        x1, y1 = outer_points_2[i]

        x2, y2 = outer_points_1[(i + 1) % 6]

        ax.plot([x1, x2],

                [y1, y2],

                color='red', linestyle='-', linewidth=3, label=f'Random Edge {i+1}-{(i+2) if (i+2) <= 6 else 1}')

  

# 设置图例和显示

# ax.legend()

plt.axis('equal')

plt.savefig('hexagonal_lattice_with_outer_layer.png')

plt.show()
