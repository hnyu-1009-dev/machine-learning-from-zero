# 导入库
# 从sklearn 导入分类的库，KNeighborsClassifier 是 KNN 算法的核心分类器
from sklearn.neighbors import KNeighborsClassifier

# 导入numpy，numpy 是用于数值计算的库，主要用于处理数组
import numpy as np

# 导入画图包，matplotlib 用于数据可视化
import matplotlib.pyplot as plt

# 目前使用课件上的数据集
# 定义三个点集合，代表不同类别的数据点，分别有两个特征（例如：坐标位置）
point1 = [
    [7.7, 6.1],  # 类别 0 的点
    [3.1, 5.9],
    [8.6, 8.8],
    [9.5, 7.3],
    [3.9, 7.4],
    [5.0, 5.3],
    [1.0, 7.3],
]
point2 = [
    [0.2, 2.2],  # 类别 1 的点
    [4.5, 4.1],
    [0.5, 1.1],
    [2.7, 3.0],
    [4.7, 0.2],
    [2.9, 3.3],
    [7.3, 7.9],
]
point3 = [
    [9.2, 0.7],  # 类别 2 的点
    [9.2, 2.1],
    [7.3, 4.5],
    [8.9, 2.9],
    [9.5, 3.7],
    [7.7, 3.7],
    [9.4, 2.4],
]

# 将数据集连接起来，将三个点集合合并成一个大的数据集
# np.concatenate: 用于连接多个数组
# axis=0 表示按行进行拼接
point_concat = np.concatenate((point1, point2, point3), axis=0)

# 设置标签，将每个数据点与其对应的类别标签进行配对
# np.zeros(len(point1)) 给 point1 的所有点分配标签 0
# np.ones(len(point2)) 给 point2 的所有点分配标签 1
# np.ones(len(point3)) + 1 给 point3 的所有点分配标签 2
point_concat_label = np.concatenate(
    (np.zeros(len(point1)), np.ones(len(point2)), np.ones(len(point3)) + 1),
    axis=0,
)

# 2. 构建KNN算法，实例化KNN算法并进行训练
# 2.1 第一步实例化KNN算法，指定邻居数量、距离度量方式等参数
n_neighbors = 5  # 确定K值（即邻居数），这里设定为5

# 使用 KNeighborsClassifier 实例化 KNN 算法模型
# n_neighbors: 选择的邻居数量，指定 KNN 算法考虑的邻居数目
# algorithm: 'brute' 表示使用暴力计算方法，即计算每个点与所有点的距离
# p: 距离度量的参数，p=2 表示使用欧几里得距离
knn = KNeighborsClassifier(
    n_neighbors=n_neighbors,  # 设置 K 值为 5
    algorithm="brute",  # 使用暴力算法来计算距离
    p=2,  # 使用欧几里得距离（p=2）
)

# 训练 KNN 模型
# knn.fit() 函数会根据训练数据（point_concat）和标签（point_concat_label）来学习
# 该函数将模型拟合到数据上，学习数据的特征与标签之间的关系
knn.fit(point_concat, point_concat_label)

# 3. 实现KNN决策边界的可视化
# 通过预测坐标网格上所有点的类别来绘制决策边界
# 获得预测点数据，使用坐标点网格来当作预测数据点
# 3.1 设定未知点，生成一个坐标点网格，表示我们希望预测的点位置
x1 = np.linspace(0, 10, 100)  # 生成从0到10的100个点，作为 x 轴的坐标
# np.linspace: 生成指定范围内的等间距数值
# 第一个参数是数据的起始值，第二个参数是结束值，第三个是生成数值的个数
y1 = np.linspace(0, 10, 100)  # 生成从0到10的100个点，作为 y 轴的坐标

# 生成坐标点网格，x_axis 和 y_axis 是二维矩阵
# meshgrid 用于生成网格坐标点，用于在平面上覆盖坐标点
x_axis, y_axis = np.meshgrid(x1, y1)

# 输出网格形状（这是一个二维的网格，用来覆盖平面）
# 网格的形状（每个轴的维度）对于决策边界的绘制非常重要
y_axis.shape

# 将 x_axis 和 y_axis 展平，转化为一维数组，以便用来做预测
# 使用 ravel() 或者 flatten() 展平 2D 数组
x_axis_ravel = x_axis.ravel()
y_axis_ravel = y_axis.ravel()

# 合并两个一维数组，得到所有网格坐标的二维形式
# np.c_[] 用来按列将两个一维数组合并成一个二维数组
xy_axis = np.c_[x_axis_ravel, y_axis_ravel]

# 4. KNN预测与绘制决策边界
# 对所有坐标点进行分类预测，knn.predict 返回的是对应点的预测标签
# knn.predict() 通过KNN模型对输入的二维数据（xy_axis）进行分类预测
knn_predict_result = knn.predict(xy_axis)

# 画图展示决策边界
# 创建一个新图形，设置图形尺寸
fig = plt.figure(figsize=(15, 20))

# 添加子图，ax 表示一个图形区域
# 111 表示 1 行 1 列的第一个图
ax = fig.add_subplot(111)

# contour: 绘制等高线，表示决策边界
# 第一个和第二个参数是坐标点网格，第三个参数是每个坐标点的预测标签
# 这里将预测结果（knn_predict_result）按网格的形状重塑，以便绘制边界
ax.contour(
    x_axis,  # x 轴坐标
    y_axis,  # y 轴坐标
    knn_predict_result.reshape(x_axis.shape),  # 将预测结果按网格形状重塑
)

# 绘制原始点的散点图
# ax.scatter 用于绘制散点，分为三个类别
# point_concat[point_concat_label == 0, 0]
# 和 point_concat[point_concat_label == 0, 1]
# 用于筛选属于类别 0 的点
ax.scatter(
    point_concat[point_concat_label == 0, 0],  # 类别 0 的 x 坐标
    point_concat[point_concat_label == 0, 1],  # 类别 0 的 y 坐标
    color="b",  # 蓝色
    marker="^",  # 使用三角形标记
)

ax.scatter(
    point_concat[point_concat_label == 1, 0],  # 类别 1 的 x 坐标
    point_concat[point_concat_label == 1, 1],  # 类别 1 的 y 坐标
    color="r",  # 红色
    marker="*",  # 使用星形标记
)

ax.scatter(
    point_concat[point_concat_label == 2, 0],  # 类别 2 的 x 坐标
    point_concat[point_concat_label == 2, 1],  # 类别 2 的 y 坐标
    color="y",  # 黄色
    marker="s",  # 使用方形标记
)

# 显示图形
# plt.show() 展示整个图形，确保图形可视化
plt.show()
