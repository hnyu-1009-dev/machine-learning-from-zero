# 导入
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 用于划分数据集
from sklearn.preprocessing import (
    StandardScaler,
)  # 使用StandardScaler可以将数据标准化为均值为0，方差为1的分布，有利于后续模型训练

# -----------------------------------------------
# 1. 数据准备：读入数据
# -----------------------------------------------

# data是pandas.DataFrame类型的二维数组
data = pd.read_excel("./Data/Real estate valuation data set.xlsx")
# 打印查看数据是否正确读入
# print(data)
# 获取列的名称
col_name = data.keys()
# print("数据处理之前的列的名称", col_name)

# -----------------------------------------------
# 2. 数据处理：为划分训练集和测试集做准备
# -----------------------------------------------

# 对便利店数量做one-hot编码处理
# 使用pd.get_dummies函数将会指定列的每个不同取值转换成一个新的二进制列
# 新的列名由原列明和取值组成
data = pd.get_dummies(
    data,
    columns=[
        "X4 number of convenience stores",
    ],
)
col_name = data.keys()
# print("数据处理之后的列名称", col_name)
# print("数据处理之后的data", data)
# 提取特征和目标变量
X = data[
    [
        "X1 transaction date",
        "X2 house age",
        "X3 distance to the nearest MRT station",
        "X5 latitude",
        "X6 longitude",
        "X4 number of convenience stores_0",
        "X4 number of convenience stores_1",
        "X4 number of convenience stores_2",
        "X4 number of convenience stores_3",
        "X4 number of convenience stores_4",
        "X4 number of convenience stores_5",
        "X4 number of convenience stores_6",
        "X4 number of convenience stores_7",
        "X4 number of convenience stores_8",
        "X4 number of convenience stores_9",
        "X4 number of convenience stores_10",
    ]
]
y = data["Y house price of unit area"]
# -----------------------------------------------
# 3. 数据集划分：将数据随机划分为训练集和测试集
# -----------------------------------------------

# 使用sklearn自带的函数去随机划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X,  # X 是特征数据
    y,  # y 是目标变量（房价）
    test_size=0.2,  # test_size=0.2 表示 20% 的数据用作测试集，80% 用作训练集
    random_state=42,  # random_state=42 确保每次运行时数据的划分是相同的
)
# -----------------------------------------------
# 4. 数据标准化并转换为tensor
# -----------------------------------------------
scaler = StandardScaler()
# !标准化的处理需要在训练集和测试集之间保持一致
X_train_scaled = scaler.fit_transform(
    X_train
)  # fit_transform() 方法是 训练数据 标准化的主要步骤。它先计算数据集的 均值（mean）和 标准差（std），然后用这些统计量对数据进行转换，将数据标准化。

X_test_scaled = scaler.transform(
    X_test
)  # transform() 方法用于 转换测试数据（或任何新的数据），不重新计算统计量，而是使用 fit() 计算得到的统计量（均值和标准差）来转换数据。

# 将处理好的数据转换成tensor,方便后续使用
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32,
)
# 获得y_train.tensor(Series类型)的值，view(-1,1) 行数自适应并将列数转换成一列
# print("y_train的值是什么", type(y_train.values))   <class 'numpy.ndarray'>
y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32,
).view(
    -1, 1
)  # ! y_train_tensor 和 y_test_tensor 需要使用 .view(-1, 1) 来确保它们的维度与模型的输出层一致
X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32,
)
y_test_tensor = torch.tensor(
    y_test.values,
    dtype=torch.float32,
).view(-1, 1)


# -----------------------------------------------
# 5. 定义线性回归模型
# -----------------------------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层，输入维度是input_size,输出维度为1，对应预测放假
        self.linear = nn.Linear(
            input_size,  # 规定了输入维度 是模型接收的输入的特征数量，即每个样本有多少个特征。就是有多少列
            1,  # 规定了输出维度
        )

    def forward(self, x):
        # 前行传播函数，定义了数据如何通过模型计算，在这里直接返回线性层的计算结果
        return self.linear(x)


# 实例化模型
model = LinearRegressionModel(X_train_tensor.shape[1])

# 定义损失函数和优化器
# 使用均方误差损失函数（MSELoss）来衡量预测和真实值之间的差异
criterion = nn.MSELoss()

# 使用Adam优化器
optimizer = optim.Adam(
    model.parameters(),  # model.parameters() 返回的是模型中 所有可训练参数（例如权重和偏置）的一个迭代器。这些参数是模型训练过程中需要优化的对象。
    lr=0.1,  # 学习率
)

# 进行模型的训练
# 设置模型训练的次数
num_epochs = 1000
for epoch in range(num_epochs):
    # 将模型设置为训练模式
    model.train()

    # 清空梯度
    optimizer.zero_grad()

    # 前向传播，的到损失
    output = model(X_train_tensor)
    loss = criterion(
        output,
        y_train_tensor,
    )

    # 根据损失值进行反向传播
    loss.backward()
    # 根据计算的到的梯度，进行更新模型的参数
    optimizer.step()

    # 设置提示信息，显示训练过程中的一些数据
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 模型评估
# model.eval()将模型设置为评估模式
model.eval()
# torch.no_grad() 是 PyTorch 中用于临时禁用梯度计算的上下文管理器，
# 主要用于推理（预测）阶段，以提高计算效率并节省内存
with torch.no_grad():
    # 将测试集的特征输入到模型中，得到预测值
    predictions = model(X_test_tensor)
    # 计算测试集上的损失，用于评估模型在未知数据上的表现
    test_loss = criterion(predictions, y_test_tensor)

# 使用matplotlib进行绘制结果
# 将tensor转换为numpy数据，方便后续绘图
predictions = predictions.detach().cpu().numpy()
y_test_numpy = y_test_tensor.detach().cpu().numpy()
# -----------------------------------------------
# 5. 定义线性回归模型
# -----------------------------------------------
# 6.画图展示
# 创建第一个图形，用于绘制散点，展示预测值和实际值的对应关系
plt.figure(0)
# 绘制散点
plt.scatter(y_test_numpy, predictions, color="blue")
# 绘制线
plt.plot(
    [min(y_test_numpy), max(y_test_numpy)],
    [min(y_test_numpy), max(y_test_numpy)],
    linestyle="--",
    color="red",
    linewidth=2,
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Regression results")

# 创建第二个图形
plt.figure(1)

# 获取测试集索引的排序后的顺序
sorted_indices = X_test.index.argsort()
# 根据排序后的索引获取对应的实际值
y_test_sorted = y_test.iloc[sorted_indices]

# 将预测值转换为Series类型，并且根据排序后的索引获取对应的值
y_pred_sorted = pd.Series(predictions.squeeze()).iloc[sorted_indices]

# 绘制实际值的曲线，用圆形标记
plt.plot(y_test_sorted.values, label="Acatual Values", marker="o")
# 绘制预测值的曲线，用*标记
plt.plot(y_pred_sorted.values, label="Predicted Values", marker="*")

# 设置轴标签和标题
plt.xlabel("Sorted Index")
plt.ylabel("Values")
plt.title("Actual vs Predicted Values in Linear Regression")
plt.show()
