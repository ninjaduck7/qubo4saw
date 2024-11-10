import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.ticker import PercentFormatter

# 读取数据
data = pd.read_csv('filling.txt', skiprows=1, header=None)
data.columns = ['scale', 'filling_rate', 'success_rate']

# 解析数据
data['n'] = data['scale'].apply(lambda x: int(x.split('x')[0]))  # 获取尺寸 n

# 分组
data_by_filling_rate = data.groupby('filling_rate')

# 拟合模型，这里假设一个简单的二次函数进行拟合
def success_rate_model(n, a, b, c):
    return a + b * n + c * n**2

plt.figure(figsize=(10, 8))

# 定义填充率到颜色的映射
filling_rate_colors = {
    0.9: 'blue',   # 90% - 蓝色
    0.8: 'orange', # 80% - 橙色
    0.7: 'green',  # 70% - 绿色
    0.6: 'red',    # 60% - 红色
    0.5: 'purple'  # 50% - 紫色
}

# 遍历每个填充率的数据组绘制图表
for filling_rate, group in data_by_filling_rate:
    n_values = group['n'].values
    success_rates = group['success_rate'].values * 100  # 转换为百分比

    # 拟合曲线
    params, _ = curve_fit(success_rate_model, n_values, success_rates, p0=[100, -1, 0.01])

    # 生成拟合曲线数据
    n_fit = np.linspace(min(n_values), max(n_values), 100)
    success_rate_fit = success_rate_model(n_fit, *params)

    # 从映射中获取颜色
    color = filling_rate_colors.get(filling_rate, 'gray')  # 默认颜色为灰色

    # 绘制散点和拟合曲线
    plt.scatter(n_values, success_rates, color=color, label=f"Data points (Filling rate: {filling_rate * 100}%)")
    plt.plot(n_fit, success_rate_fit, color=color, linestyle='--', label=f"Fitted curve (Filling rate: {filling_rate * 100}%)")

# 设置图表信息
plt.xlabel("Scale (Width = Height = n)")
plt.ylabel("Success rate (%)")
plt.title("Scale vs Success rate at Different Filling rates")
plt.gca().yaxis.set_major_formatter(PercentFormatter())  # 设置 Y 轴为百分比格式
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig("scale_vs_success_rate_multiple_filling.png", format='png', dpi=300)
# plt.show()