import random
import time

# 参数设置
target_filling_rate = 0.5 # 期望填充率
total_runs = 2000  # 总运行次数

# 执行不同尺度的运行
for n in range(3, 11):  # 从 3 到 10 的不同尺度
    width = height = n
    tmax = int(target_filling_rate * width * height)  # 计算每个尺度下的 tmax
    times = []  # 记录每次运行的时间
    successful_runs = 0  # 记录成功次数

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

        # 检查是否走满了 tmax 步
        if valid_run and len(x) == tmax:
            successful_runs += 1  # 计为有效运行

    # 计算平均时间和成功率
    average_time = sum(times) / len(times)
    success_rate = successful_runs / total_runs

    # 保存时间记录和成功率
    with open("time.txt", "a") as f:
        f.write(f"\n--- Width = Height = {n} ---\n")
        f.write("Run times:\n")
        for i, t in enumerate(times, 1):
            f.write(f"Run {i}: {t:.4f} seconds\n")
        f.write(f"\nAverage time: {average_time:.4f} seconds\n")
        f.write(f"Success rate: {success_rate:.2%} ({successful_runs}/{total_runs})\n")

    # 记录填充率和成功率到 filling.txt
    with open("filling.txt", "a") as f:
        f.write(f"Width: {width}, Height: {height}, Target Filling-rate: {target_filling_rate:.2%}, Calculated tmax: {tmax}, Success rate: {success_rate:.2%}\n")

    print(f"Scale {n} completed. Average Runtime: {average_time:.4f} seconds")
    print(f"Success rate: {success_rate:.2%} ({successful_runs}/{total_runs})")
    print(f"Filling rate for {n} recorded in filling.txt")