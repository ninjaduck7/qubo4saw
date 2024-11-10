from pyqubo import Binary,Constraint
import neal
from hexaplot import draw_hexagonal_system_and_save
import os


# 确保存储图像的文件夹存在
output_dir = "pic"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# 定义边的二进制变量
edges = {
    # 内层六边形的六条边
    "x_inner_12": Binary("x_inner_12"),
    "x_inner_23": Binary("x_inner_23"),
    "x_inner_34": Binary("x_inner_34"),
    "x_inner_45": Binary("x_inner_45"),
    "x_inner_56": Binary("x_inner_56"),
    "x_inner_61": Binary("x_inner_61"),

    # 向外延伸的六条边（tunnel）
    "x_tunnel_1": Binary("x_tunnel_1"),
    "x_tunnel_2": Binary("x_tunnel_2"),
    "x_tunnel_3": Binary("x_tunnel_3"),
    "x_tunnel_4": Binary("x_tunnel_4"),
    "x_tunnel_5": Binary("x_tunnel_5"),
    "x_tunnel_6": Binary("x_tunnel_6"),

    # 外层与tunnel相交的12条边
    "x_1_outer_1": Binary("x_1_outer_1"),
    "x_1_outer_2": Binary("x_1_outer_2"),
    "x_2_outer_1": Binary("x_2_outer_1"),
    "x_2_outer_2": Binary("x_2_outer_2"),
    "x_3_outer_1": Binary("x_3_outer_1"),
    "x_3_outer_2": Binary("x_3_outer_2"),
    "x_4_outer_1": Binary("x_4_outer_1"),
    "x_4_outer_2": Binary("x_4_outer_2"),
    "x_5_outer_1": Binary("x_5_outer_1"),
    "x_5_outer_2": Binary("x_5_outer_2"),
    "x_6_outer_1": Binary("x_6_outer_1"),
    "x_6_outer_2": Binary("x_6_outer_2"),

    # 最外层连接的六条边（随机连接）
    "x_random_12": Binary("x_random_12"),
    "x_random_23": Binary("x_random_23"),
    "x_random_34": Binary("x_random_34"),
    "x_random_45": Binary("x_random_45"),
    "x_random_56": Binary("x_random_56"),
    "x_random_61": Binary("x_random_61"),

    #引入辅助变量
    "y_tunnel_outer_1": Binary("y_tunnel_outer_1"),
    "y_tunnel_outer_2": Binary("y_tunnel_outer_2"),
    "y_tunnel_outer_3": Binary("y_tunnel_outer_3"),
    "y_tunnel_outer_4": Binary("y_tunnel_outer_4"),
    "y_tunnel_outer_5": Binary("y_tunnel_outer_5"),
    "y_tunnel_outer_6": Binary("y_tunnel_outer_6"),

    "y_tunnel_inner_1": Binary("y_tunnel_inner_1"),
    "y_tunnel_inner_2": Binary("y_tunnel_inner_2"),
    "y_tunnel_inner_3": Binary("y_tunnel_inner_3"),
    "y_tunnel_inner_4": Binary("y_tunnel_inner_4"),
    "y_tunnel_inner_5": Binary("y_tunnel_inner_5"),
    "y_tunnel_inner_6": Binary("y_tunnel_inner_6"),

}

# 设置总数 N
N = 15 # 你可以自由设置的总数
num_reads = 10
# 计算前 30 个 qubit 变量的和
sum_of_qubits = sum(list(edges.values())[:30])

# 定义哈密顿量 H_walk，仅使用前 30 个 qubit
H_walk = Constraint((sum_of_qubits - N) ** 2, label="H_walk")

# 定义 H_tunnel_outer

# 引入辅助qubits http://arxiv.org/abs/1307.8041
H_tunnel_outer = Constraint(

   (Binary('y_tunnel_outer_1')*Binary('x_tunnel_1') +
 2*(3*Binary('y_tunnel_outer_1') +
    Binary('x_1_outer_1') * Binary('x_1_outer_2') -
    2*Binary('x_1_outer_1')*Binary('y_tunnel_outer_1') -
    2*Binary('x_1_outer_2')*Binary('y_tunnel_outer_1'))) +

(Binary('y_tunnel_outer_2')*Binary('x_tunnel_2') +
 2*(3*Binary('y_tunnel_outer_2') +
    Binary('x_2_outer_1') * Binary('x_2_outer_2') -
    2*Binary('x_2_outer_1')*Binary('y_tunnel_outer_2') -
    2*Binary('x_2_outer_2')*Binary('y_tunnel_outer_2'))) +

(Binary('y_tunnel_outer_3')*Binary('x_tunnel_3') +
 2*(3*Binary('y_tunnel_outer_3') +
    Binary('x_3_outer_1') * Binary('x_3_outer_2') -
    2*Binary('x_3_outer_1')*Binary('y_tunnel_outer_3') -
    2*Binary('x_3_outer_2')*Binary('y_tunnel_outer_3'))) +

(Binary('y_tunnel_outer_4')*Binary('x_tunnel_4') +
 2*(3*Binary('y_tunnel_outer_4') +
    Binary('x_4_outer_1') * Binary('x_4_outer_2') -
    2*Binary('x_4_outer_1')*Binary('y_tunnel_outer_4') -
    2*Binary('x_4_outer_2')*Binary('y_tunnel_outer_4'))) +

(Binary('y_tunnel_outer_5')*Binary('x_tunnel_5') +
 2*(3*Binary('y_tunnel_outer_5') +
    Binary('x_5_outer_1') * Binary('x_5_outer_2') -
    2*Binary('x_5_outer_1')*Binary('y_tunnel_outer_5') -
    2*Binary('x_5_outer_2')*Binary('y_tunnel_outer_5'))) +

(Binary('y_tunnel_outer_6')*Binary('x_tunnel_6') +
 2*(3*Binary('y_tunnel_outer_6') +
    Binary('x_6_outer_1') * Binary('x_6_outer_2') -
    2*Binary('x_6_outer_1')*Binary('y_tunnel_outer_6') -
    2*Binary('x_6_outer_2')*Binary('y_tunnel_outer_6')))
,
    label="H_tunnel_outer"
)

H_tunnel_inner = Constraint(

(Binary('y_tunnel_inner_1')*Binary('x_tunnel_1') +
 2*(3*Binary('y_tunnel_inner_1') +
    Binary('x_inner_12') * Binary('x_inner_61') -
    2*Binary('x_inner_12')*Binary('y_tunnel_inner_1') -
    2*Binary('x_inner_61')*Binary('y_tunnel_inner_1'))) +

(Binary('y_tunnel_inner_2')*Binary('x_tunnel_2') +
 2*(3*Binary('y_tunnel_inner_2') +
    Binary('x_inner_23') * Binary('x_inner_12') -
    2*Binary('x_inner_23')*Binary('y_tunnel_inner_2') -
    2*Binary('x_inner_12')*Binary('y_tunnel_inner_2'))) +

(Binary('y_tunnel_inner_3')*Binary('x_tunnel_3') +
 2*(3*Binary('y_tunnel_inner_3') +
    Binary('x_inner_34') * Binary('x_inner_23') -
    2*Binary('x_inner_34')*Binary('y_tunnel_inner_3') -
    2*Binary('x_inner_23')*Binary('y_tunnel_inner_3'))) +

(Binary('y_tunnel_inner_4')*Binary('x_tunnel_4') +
 2*(3*Binary('y_tunnel_inner_4') +
    Binary('x_inner_45') * Binary('x_inner_34') -
    2*Binary('x_inner_45')*Binary('y_tunnel_inner_4') -
    2*Binary('x_inner_34')*Binary('y_tunnel_inner_4'))) +

(Binary('y_tunnel_inner_5')*Binary('x_tunnel_5') +
 2*(3*Binary('y_tunnel_inner_5') +
    Binary('x_inner_56') * Binary('x_inner_45') -
    2*Binary('x_inner_56')*Binary('y_tunnel_inner_5') -
    2*Binary('x_inner_45')*Binary('y_tunnel_inner_5'))) +

(Binary('y_tunnel_inner_6')*Binary('x_tunnel_6') +
 2*(3*Binary('y_tunnel_inner_6') +
    Binary('x_inner_61') * Binary('x_inner_56') -
    2*Binary('x_inner_61')*Binary('y_tunnel_inner_6') -
    2*Binary('x_inner_56')*Binary('y_tunnel_inner_6')))
,
    label="H_tunnel_inner"
)




H = H_walk

H= H+100* H_tunnel_outer + 100*H_tunnel_inner
model = H.compile()
bqm = model.to_bqm()

# Solve using simulated annealing
sa = neal.SimulatedAnnealingSampler()
sampleset = sa.sample(bqm, num_reads=num_reads)




# 设置保存图片的文件夹路径
save_dir = "./pic"

# 如果文件夹不存在则创建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Example usage:
binary_strings = []  # 用来保存所有的前30个bit
for i, sample in enumerate(sampleset.record['sample']):
    # 提取前30个bit
    binary_string = ''.join(map(str, sample[:30]))
    binary_strings.append(binary_string)  # 将每次的结果存入列表中
    
    # 生成保存图片的文件名并调用绘图函数保存图片
    save_path = os.path.join(save_dir, f"hex_system_{i}.png")
    draw_hexagonal_system_and_save(binary_string, save_path=save_path)

# 将每次采样的结果保存到文件
with open("sampling_results.txt", "w") as f:
    for i, sample in enumerate(sampleset):
        f.write(f"Sample {i + 1}:\n")  # 显示采样的编号
        for edge_name in edges.keys():
            f.write(f"  {edge_name}: {sample[edge_name]}\n")
        f.write("\n")





