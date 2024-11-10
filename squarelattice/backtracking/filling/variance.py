import re
import pandas as pd

# 从文件读取数据并解析
data = []
with open("filling.txt", "r") as file:
    for line in file:
        match = re.search(r"Width: (\d+), Height: (\d+), Target Filling-rate: ([\d.]+)%, .* Success rate: ([\d.]+)%", line)
        if match:
            n = int(match.group(1))  # Width or Height (assume square grid)
            filling_rate = float(match.group(3))  # Filling rate
            success_rate = float(match.group(4)) / 100  # 将 success_rate 转换为小数形式
            data.append((n, filling_rate, success_rate))

# 将数据转换为 DataFrame 以便于处理
df = pd.DataFrame(data, columns=['scale', 'filling_rate', 'success_rate'])

# 计算每个 scale 下不同 filling_rate 的 success_rate 方差
variance_data = df.groupby('scale')['success_rate'].var().reset_index()
variance_data['variance'] = variance_data['success_rate'] * 100  # 将方差结果转换为百分数形式
variance_data = variance_data[['scale', 'variance']]  # 仅保留所需列

# 将结果保存到 variance.txt 文件中
with open("variance.txt", "w") as f:
    f.write("Scale, Variance (%)\n")
    for _, row in variance_data.iterrows():
        f.write(f"{row['scale']}, {row['variance']:.2f}%\n")

print("Variance calculation completed and saved to variance.txt.")