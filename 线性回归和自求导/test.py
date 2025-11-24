# 导入包
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec   # gridspec 用于自定义网格布局

# -------------------------
# 1. 数据准备：散点输入
# -------------------------

data = [[-0.5, 7.7],
        [1.8, 98.5],
        [0.9, 57.8],
        [0.4, 39.2],
        [-1.4, -15.7],
        [-1.4, -37.3],
        [-1.8, -49.1],
        [1.5, 75.6],
        [0.4, 34],
        [0.8, 62.3]]

# 转成 numpy 数组，便于后续的矩阵运算
data = np.array(data)

# 按列提取 x 和 y
# data[:,0] 代表第一列 → x 值
# data[:,1] 代表第二列 → y 值
x_data = data[:, 0]
y_data = data[:, 1]

# -------------------------
# 2. 参数初始化
# -------------------------

w = 0   # 权重（斜率）
b = 0   # 偏置（截距）

learning_rate = 0.01  # 超参数：学习率，控制每次参数更新幅度

# -------------------------
# 3. 损失函数（均方误差 MSE）
# -------------------------

def loss_function(x_data,  # 输入数据的 x（自变量）
                  y_data,  # 输入数据的 y（真实值 / 标签）
                  w,       # 参数：权重（斜率）
                  b        # 参数：偏置（截距）
                 ):
    """
    使用均方误差（MSE）计算线性回归模型的损失值。
    即：Loss = mean( (真实值 - 预测值)^2 )
    """

    # 预测值计算：
    # np.dot(x_data, w) 表示所有 x 与 w 做点积（线性模型）
    # 加上偏置 b 得到预测结果 predicted
    predicted = np.dot(x_data, w) + b

    # 均方误差 MSE：
    # (y_data - predicted)**2 → 每个样本点的误差平方
    # np.mean(...) → 所有误差平方的平均值
    total_loss = np.mean((y_data - predicted) ** 2)

    # 返回标量损失，用于衡量当前 w,b 的好坏
    return total_loss


# ==========================================================
# 下面开始画图（这部分不做注释，按你的要求保持原样）
# ==========================================================

fig = plt.figure("show figure", figsize=(12, 6))
gs = gridspec.GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0,0])
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('figure1 data')

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_xlabel('iter')
ax2.set_ylabel('e')
ax2.set_title('figure2 data')

ax3=fig.add_subplot(gs[0,1],projection='3d')
w_values = np.linspace(-20, 80, 100)
b_values = np.linspace(-20, 80, 100)
W, B = np.meshgrid(w_values, b_values)
loss_values=np.zeros_like(W)

for i ,w_value in enumerate(w_values):
    for j, b_value in enumerate(b_values):
        loss_values[j, i] = loss_function(x_data, y_data, w_value, b_value)

ax3.plot_surface(W, B, loss_values, cmap='viridis', alpha=0.8)
ax3.set_xlabel('w')
ax3.set_ylabel('b')
ax3.set_zlabel('loss')
ax3.set_title('figure3  surface plot')

ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlabel('w')
ax4.set_ylabel('b')
ax4.set_title('coutour plot')
ax4.contourf(W, B, loss_values, levels=20, cmap='viridis')


# ==========================================================
# 迭代训练（梯度下降部分）：你要求注释的重点
# ==========================================================

num_iterations = 100  # 梯度下降的迭代次数

iter_loss_list = []   # 存储每轮损失值，用作绘制损失下降曲线
iter_num_list = []    # 存储迭代轮数
gd_path = []          # 存储每次 (w,b) 的位置，用于 3D 轨迹显示

for n in range(1, num_iterations + 1):

    gd_path.append((w, b))  # 保存当前参数点，用于轨迹图

    # -------- 前向传播：计算预测值 --------
    y_pre = np.dot(x_data, w) + b

    # -------- 计算当前损失 --------
    e = np.mean((y_data - y_pre) ** 2)

    iter_loss_list.append(e)   # 保存损失
    iter_num_list.append(n)    # 保存迭代次数

    # -------- 反向传播：计算梯度 --------
    # 梯度公式（对 w 求偏导）：
    # dL/dw = -2/N * (y - y_pre) dot x_data
    gradient_w = (-2 * (y_data - y_pre).dot(x_data)) / len(x_data)

    # 梯度公式（对 b 求偏导）：
    # dL/db = -2/N * sum(y - y_pre)
    gradient_b = np.mean(-2 * (y_data - y_pre))

    # -------- 更新参数：梯度下降 --------
    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

    # -------- 显示更新（画图相关，保持不注释）--------
    frequence_display = 10
    if n % frequence_display == 0 or n == 1:
        ax1.clear()
        ax1.set_xlabel('X')
        ax1.set_ylabel('y')
        ax1.set_title('figure1 data')
        ax1.scatter(x_data, y_data, color='b')
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = w * x_min + b, w * x_max + b
        ax1.plot([x_min, x_max], [y_min, y_max], color='r')
        
        ax2.clear()
        ax2.set_xlabel('iter')
        ax2.set_ylabel('e')
        ax2.set_title('figure2 data')
        ax2.plot(iter_num_list, iter_loss_list, color='g')

        if len(gd_path) > 0:
            gd_w, gd_b = zip(*gd_path)
            ax3.plot(gd_w, gd_b,
                    [loss_function(x_data, y_data, np.array(gd_w[i]), np.array(gd_b[i])) for i in range(len(gd_w))],
                    color='black')
            ax3.set_xlim(-20, 80)
            ax3.set_ylim(-20, 80)
            ax3.set_xlabel('w')
            ax3.set_ylabel('b')
            ax3.set_zlabel('loss')
            ax3.set_title('figure3  surface plot')
            ax3.scatter(w, b, loss_function(x_data, y_data, w, b), c='black', s=20)

            ax4.plot(gd_w, gd_b)

        plt.pause(1)
