import pandas as pd

# 读取数据
data = pd.read_csv('filling.txt', skiprows=1, header=None)
data.columns = ['scale', 'filling_rate', 'success_rate']

# 解析数据
data['n'] = data['scale'].apply(lambda x: int(x.split('x')[0]))  # 获取尺寸 n

# 分组计算方差并重命名方差列
variance_data = data.groupby('n')['success_rate'].var().reset_index(name='variance')
variance_data['variance'] = variance_data['variance'] * 100  # 转换为百分数

# 将结果保存到 variance.txt 文件中
with open('variance.txt', 'w') as f:
    f.write("Scale, Variance (%)\n")
    for _, row in variance_data.iterrows():
        f.write(f"{row['n']}, {row['variance']:.2f}%\n")

print("Variance calculation completed and saved to variance.txt.")