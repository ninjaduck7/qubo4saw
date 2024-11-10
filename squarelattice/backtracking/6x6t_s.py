import random
import os
import time

# 设置最大步数和格子大小
tmax = 20
width = height = 5

# 准备记录时间
times = []
successful_runs = 0  # 记录有效运行次数
total_runs = 2000  # 总共运行次数

for run in range(total_runs):
    start_time = time.time()
    
    x = [0]
    y = [0]
    coodr = [[x[0], y[0]]]
    valid_run = True

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
            valid_run = False  # 提前结束，标记为无效
            break

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

    # 记录此次运行时间
    end_time = time.time()
    run_time = end_time - start_time
    times.append(run_time)

    # 检查是否走满了20步
    if valid_run and len(x) == tmax:
        successful_runs += 1  # 计为有效运行

# 计算平均时间
average_time = sum(times) / len(times)

# 计算成功率
success_rate = successful_runs / total_runs

# 保存时间记录和成功率
with open("time.txt", "w") as f:
    f.write("Run times:\n")
    for i, time in enumerate(times, 1):
        f.write(f"Run {i}: {time:.4f} seconds\n")
    f.write(f"\nAverage time: {average_time:.4f} seconds\n")
    f.write(f"Success rate: {success_rate:.2%} ({successful_runs}/{total_runs})\n")

print(f"All runs completed. Average Runtime: {average_time:.4f} seconds")
print(f"Success rate: {success_rate:.2%} ({successful_runs}/{total_runs})")