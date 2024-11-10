import random
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# 创建图片存储目录
pic_folder = "pic"
if not os.path.exists(pic_folder):
    os.makedirs(pic_folder)

# 设置最大步数和格子大小
tmax = 30
width = height = 6

# 准备记录时间
times = []

for run in range(20):
    start_time = time.time()
    
    x = [0]
    y = [0]
    coodr = [[x[0], y[0]]]

    for t in range(1, tmax):
        available_dir = []
        if (x[t-1] > 0) and [(x[t-1] - 1), y[t-1]] not in coodr:
            available_dir.append("left")
        if (x[t-1] < width - 1) and [(x[t-1] + 1), y[t-1]] not in coodr:
            available_dir.append("right")
        if (y[t-1] > 0) and [x[t-1], (y[t-1] - 1)] not in coodr:
            available_dir.append("down")
        if (y[t-1] < height - 1) and [x[t-1], (y[t-1] + 1)] not in coodr:
            available_dir.append("up")

        if not available_dir:
            break  # 如果没有可用方向，提前结束

        chosen_dir = random.choice(available_dir)
        if chosen_dir == "left":
            x.append(x[t-1] - 1)
            y.append(y[t-1])
        elif chosen_dir == "right":
            x.append(x[t-1] + 1)
            y.append(y[t-1])
        elif chosen_dir == "down":
            x.append(x[t-1])
            y.append(y[t-1] - 1)
        elif chosen_dir == "up":
            x.append(x[t-1])
            y.append(y[t-1] + 1)
        
        coodr.append([x[t], y[t]])

    # 绘图
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlim(0, width - 1)
    plt.ylim(0, height - 1)
    plt.grid(True)
    plt.title(f"SAW Run {run+1} (tmax={tmax})")  # 标题中添加 tmax
    plt.savefig(os.path.join(pic_folder, f"saw_run_{run+1}.png"))
    plt.close()

    # 记录此次运行时间
    end_time = time.time()
    run_time = end_time - start_time
    times.append(run_time)

# # 计算平均时间
# average_time = sum(times) / len(times)

# # 保存时间记录
# with open("time.txt", "w") as f:
#     f.write("Run times:\n")
#     for i, time in enumerate(times, 1):
#         f.write(f"Run {i}: {time} seconds\n")
#     f.write(f"Average time: {average_time} seconds\n")

# # 该脚本将不会打开图片窗口
# print(f"All runs completed. Average Runtime: {average_time} seconds")
