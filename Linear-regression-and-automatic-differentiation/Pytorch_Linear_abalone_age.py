import pandas as pd
from openpyxl.styles.builtins import output
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------------------------
# 1. 加载数据集
# -----------------------------------------------
# 使用pd.read_csv读取abalone.data文件，指定分隔符为逗号
data = pd.read_csv("./Data/abalone/abalone.data", sep=",")
# 打印前5行数据初步查看数据内容和格式
# print("打印数据：\n", data)
# -----------------------------------------------
# 2. 数据处理
# -----------------------------------------------
# 2.1因为ablone.data数据中没有列名，添加列名
# 定义一个包含各列名称的列表，为数据框中的列赋予有意义的名称
# 使得后续数据处理和分析时更清晰明确各列代表的含义
# 2.1先定义和数据列对应的列名(列名的个数和数据列对应)
column_names = [
    "Sex",  # 性别
    "Length",  # 长度
    "Diameter",  # 直径
    "Height",  # 高度
    "Whole_weight",  # 整体重量
    "Shucked_weight",  # 去壳重量
    "Viscera_weight",  # 内脏重量
    "Shell_weight",  # 壳重
    "Rings",  # 环数
]
# 2.2将column_names添加到数据上，目的为后续数据处理和分析时更清晰明确各列代表的含义
data.columns = column_names
# print("打印添加列之后的前5行数据：\n", data.head(5))
# 2.2.对表示类别没有大小关系的  'Sex' 列进行one-hot编码
# 因为 'Sex' 列是分类变量（包含不同的性别类别），one-hot编码会将其转换为多个二进制列
# 例如原本的 'Sex' 列有 'F'、'M'、'I' 等类别，编码后会生成 'Sex_F'、'Sex_M'、'Sex_I' 等新列，方便模型处理分类特征
# 对性别做one-hot编码处理
# 使用pd.get_dummies函数将会指定列的每个不同取值转换成一个新的二进制列
# 新的列名由原列明和性别代表字母组成
data = pd.get_dummies(data, columns=["Sex"])
# print(data.keys())
# 2.3.提取特征和目标变量
# 从处理后的数据框中选取要作为模型输入特征的列，组成特征矩阵X
X = data[
    [
        "Sex_F",
        "Sex_M",
        "Sex_I",
        "Length",
        "Diameter",
        "Height",
        "Whole_weight",
        "Shucked_weight",
        "Viscera_weight",
        "Shell_weight",
    ]
]
# 选取 'Rings' 列作为目标变量，即模型要预测的对象，通常代表了鲍鱼的年龄相关信息
y = data["Rings"]
# -----------------------------------------------
# 3. 将数据分为训练集和测试集
# -----------------------------------------------
# 使用sklearn的train_test_split函数按照指定的测试集比列(test_size=0.2 表示20%的数据作为测试集)
# 随机种子(random_state=42,保证每次划分结果的一致，便于复现和对比不同的实验情况)来划分数据集
# 划分后得到训练集特征X_train、测试集特征X_test、训练集目标y_train、测试集目标y_test


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# -----------------------------------------------
# 4. 将数据分进行标准化并转换为tensor
# -----------------------------------------------
# 创建StandardScaler对象，用于对数据进行标准化处理，使其具有均值为0，方差为1的分布特点
# 先在训练集上拟合标准化器（计算均值、方差等统计量），然后用拟合好的标准化器对训练集和测试集分别进行转换
scaler = StandardScaler()
# scaler.fit_transform用于训练数据，先计算统计量再转换数
X_train_scaled = scaler.fit_transform(X_train)
# scaler.transform 用于测试数据或新数据，仅转换数据（使用训练数据的统计量）
X_test_scaled = scaler.transform(X_test)
# 将数据转换为 PyTorch 张量
# 将标准化后的训练集特征、训练集目标、测试集特征、测试集目标都转换为PyTorch的张量格式
# 并指定数据类型为float32，这是因为PyTorch模型中的计算通常要求数据为特定的张量类型，方便后续运算
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# -----------------------------------------------
# 5. 定义回归模型并进行训练
# -----------------------------------------------
# 定义线性回归模型
# 自定义的线性回归模型类，继承自nn.Module，这是PyTorch中构建神经网络模型的基类
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        # 定义一个线性层，输入维度为input_size（即特征数量），输出维度为1，对应要预测的目标变量维度（这里是预测的环数）
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


# 实例化模型
# 根据训练集特征张量的列数确定输入维度，即特征数，以此实例化线性回归模型
input_size = X_train_tensor.shape[1]
model = LinearRegression(input_size)

# 定义损失函数和优化器
# 使用均方误差损失函数（MSELoss），他可以衡量预测值和真实值之间的差异程度，是回归问题常用的损失函数
criterion = nn.MSELoss()
# 使用Adam优化器，传入模型的可学习参数（也就是模型中线性层的权重等参数），并指定学习率为0.1
# 优化器的作用是根据损失函数计算得到的梯度信息来更新模型参数，以尝试减小损失函数的值
optimizer = optim.Adam(
    model.parameters(),
    lr=0.03,
)
# 定义学习率调度器
scheduler = StepLR(
    optimizer,
    step_size=200,
    gamma=0.3,
)
# 使用DataLoader进行批量训练
train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
# 训练模型
num_epochs = 5000
for epochs in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        # 前向传播
        # 将训练集特征张量传入模型，通过模型的前向传播计算得到预测值
        outputs = model(batch_x)
        # 计算预测值和真实训练集目标之间的损失，调用之前定义的损失函数来完成计算
        loss = criterion(
            outputs,
            batch_y,
        )
        # 反向传播和优化
        # 计算损失关于模型参数的梯度，Pytorch会自动根据构建的计算图来完成这一复杂的求导过程
        loss.backward()
        # 根据计算得到的梯度，使用优化器来更新模型的阐述，按照学习率等设置调整参数，使得 模型朝着损失值减小的方向优化
        # 迭代并优化参数
        optimizer.step()

    # 更新学习率
    scheduler.step()
    # 每 1000 个 epoch 打印一次损失
    if (epochs + 1) % 1000 == 0:
        print(
            f"Epoch [{epochs+1}/{num_epochs}], Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}"
        )
        if loss.item() < 2:
            break
    # 获取当前的学习率
    current_lr = scheduler.get_last_lr()[0]
    print("current_lr", current_lr)
    if current_lr <= 0:
        break
# 评估模型
model.eval()
with torch.no_grad():
    # 在测试集上进行预测，将测试集特征张量传入模型得到预测值
    predictions = model(X_test_tensor)
    # 计算测试集上的损失，用于评估模型在未见过的数据(测试集) 上的表现，同样调用损失函数来计算
    test_loss = criterion(
        predictions,
        y_test_tensor,
    )
    print("test_loss", test_loss.item())

# 将测试值和目标转换为Numpy数组
# 把Pytorch张量形式的预测值和测试集转换为Numpy数组，方便后续画图
predictions = predictions.detach().cpu().numpy()
y_test_numpy = y_test_tensor.detach().cpu().numpy()


# -----------------------------------------------
# 6. 绘制图表
# -----------------------------------------------
# 使用matplotlib绘制散点图展示预测值和实际值的关系
# 蓝色散点表示实际值和预测值对应的各个点，直观呈现模型预测效果与实际情况的差异
import matplotlib.pyplot as plt

plt.figure(0)
plt.scatter(y_test_numpy, predictions, color="blue")
# 绘制一条对角线（虚线红色，线宽为2），代表理想情况下预测值和实际值完全相等的情况，用于对比参考
plt.plot(
    [min(y_test_numpy), max(y_test_numpy)],
    [min(y_test_numpy), max(y_test_numpy)],
    linestyle="--",
    color="red",
    linewidth=2,
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Regression Results")
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
