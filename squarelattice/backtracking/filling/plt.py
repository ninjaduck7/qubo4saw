import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.optimize import curve_fit

# 定义拟合函数（例如指数衰减模型）
def success_rate_model(n, a, b, c):
    return a * np.exp(-b * n) + c

# 准备一个存储不同填充率数据的字典
data_by_filling_rate = {}

# 从文件读取数据，按照填充率分类
with open("filling.txt", "r") as file:
    for line in file:
        match = re.search(r"Width: (\d+), Height: (\d+), Target Filling-rate: ([\d.]+)%, .* Success rate: ([\d.]+)%", line)
        if match:
            n = int(match.group(1))  # Width or Height (assume square grid)
            filling_rate = float(match.group(3))  # Filling rate
            success_rate = float(match.group(4))
            
            # 将数据按填充率存入字典
            if filling_rate not in data_by_filling_rate:
                data_by_filling_rate[filling_rate] = []
            data_by_filling_rate[filling_rate].append((n, success_rate))

# 绘制不同填充率的曲线
plt.figure(figsize=(10, 8))
for filling_rate, data in data_by_filling_rate.items():
    data = np.array(data)
    n_values = data[:, 0]
    success_rates = data[:, 1]
    
    # 拟合曲线
    params, _ = curve_fit(success_rate_model, n_values, success_rates, p0=[100, 0.1, 0])
    
    # 生成拟合曲线
    n_fit = np.linspace(min(n_values), max(n_values), 100)
    success_rate_fit = success_rate_model(n_fit, *params)
    
    # 绘制散点和拟合曲线
    plt.scatter(n_values, success_rates, label=f"Data points (Filling-rate: {filling_rate}%)")
    plt.plot(n_fit, success_rate_fit, linestyle='--', label=f"Fitted curve (Filling-rate: {filling_rate}%)")

# 设置图表信息
plt.xlabel("Scale (Width = Height = n)")
plt.ylabel("Success rate (%)")
plt.title("Scale vs Success rate at Different Filling-rates(Blind ants)")
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig("scale_vs_success_rate_multiple_filling.png", format='png', dpi=300)
# plt.show()