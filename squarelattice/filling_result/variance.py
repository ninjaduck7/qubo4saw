import matplotlib.pyplot as plt

# 定义 QUBO 数据
qubo_data = {
    "Scale": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "Variance": [0.08, 0.03, 1.22, 0.59, 0.25, 0.37, 0.13, 0.06]  # 已转换为百分比
}

# 定义 Blind-ant 数据
blind_ant_data = {
    "Scale": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    "Variance": [1.76, 3.77, 6.62, 5.19, 3.52, 1.76, 0.86, 0.33]
}

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制 QUBO 数据折线图
plt.plot(qubo_data["Scale"], qubo_data["Variance"], label="QUBO", marker='o', color="blue", linestyle='-', linewidth=2)

# 绘制 Blind-ant 数据折线图
plt.plot(blind_ant_data["Scale"], blind_ant_data["Variance"], label="Blind-ant", marker='s', color="green", linestyle='--', linewidth=2)

# 美化图表
plt.xlabel("Scale (Width = Height = n)", fontsize=12)
plt.ylabel("Variance (%) (filling-rate 0.5--0.9)", fontsize=12)
plt.title("Variance Comparison Between QUBO and Blind-ant", fontsize=14, fontweight='bold')
plt.legend(title="Method", fontsize=10, title_fontsize='13')
plt.grid(True, linestyle='--', alpha=0.7)

# 设置坐标轴刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 保存并显示图像
plt.savefig("variance_comparison_qubo_blind_ant.png", format='png', dpi=300)
# plt.show()