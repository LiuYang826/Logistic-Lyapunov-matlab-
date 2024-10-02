# -*- ecoding: utf-8 -*-
# @Author: Lee
# @Time: 2024/9/14 10:13
import numpy as np
import matplotlib.pyplot as plt


# Logistic映射定义
def logistic_map(n, a):
    return a * n * (1 - n)


# Logistic映射导数
def derivative(n, a):
    return - 2 * a * n


# iterations迭代次数，transitory暂时忽略前100次迭代以消除初始条件的影响
def compute_lyapunov(a, iterations=1000, transitory=100):
    n = 0.1  # 初始条件0-1
    lyapunov_sum = 0  # 初始化Lyapunov指数

    for i in range(iterations + transitory):
        # 驰豫时对映射进行足够多的迭代，以使系统达到其不变分布
        if i >= transitory:
            # 添加了1e-10这个小的常数，它确保了即使导数为零，对数函数也不会遇到除以零的情况
            lyapunov_sum += np.log(abs(derivative(n, a)) + 1e-10)
        # 更新n值
        n = logistic_map(n, a)
    # 计算平均Lyapunov指数
    return lyapunov_sum / (iterations)


alpha = np.arange(2.75, 4.01, 0.01)
# 初始化Lyapunov指数数组
lyapunov_exponents = []

for a in alpha:
    lyapunov_exponents.append(compute_lyapunov(a))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
# 初始化Lyapunov指数数组
plt.plot(alpha, lyapunov_exponents, label='Lyapunov 指数')
plt.title('Logistic映射的李雅普诺夫指数')
plt.xlabel('α')
plt.ylabel('Lyapunov 指数')
plt.legend()
plt.show()

# result:当alph大于3.6的时候，系统已经要进入混沌态了