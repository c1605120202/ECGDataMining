import multiprocessing

# 目标函数：计算平方
def compute_square(n):
    print(f"计算 {n} 的平方")
    return n * n

def main():
    numbers = [i for i in range(1, 6)]  # 示例数据
    num_processes = multiprocessing.cpu_count()  # 使用的进程数

    # 使用进程池，确保结果按输入顺序输出
    with multiprocessing.Pool(processes=num_processes) as pool:
        result = pool.map(compute_square, numbers)  # map 保证输出顺序与输入顺序一致

    print("计算结果:", result)

if __name__ == '__main__':
    main()