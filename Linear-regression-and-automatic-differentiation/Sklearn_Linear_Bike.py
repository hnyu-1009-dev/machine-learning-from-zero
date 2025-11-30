import pandas as pd

# -----------------------------------------------
# 1. 加载数据集
# -----------------------------------------------
# 使用 pandas 的 read_csv 函数读取数据集，数据分隔符为逗号（CSV 格式）
dataset = pd.read_csv("./Data/bike+sharing+dataset/day.csv", sep=",")
# 打印数据的前几行（默认输出前5行），查看数据结构，列名及数据类型等信息
print(dataset.head())

# -----------------------------------------------
# 2. 数据预处理
# -----------------------------------------------
# 获取特征矩阵 X，排除目标变量 'cnt'、'instant' 和 'dteday' 三列
# 'cnt' 是目标变量，我们要预测的内容；'instant' 和 'dteday' 是无意义的索引列
X = dataset.drop(["cnt", "instant", "dteday"], axis=1)
# 获取目标变量 y（即 'cnt' 列），这是我们要预测的值
y = dataset["cnt"]

# 处理类别特征：进行 One-Hot 编码
# 定义一个列表，包含所有需要 One-Hot 编码的类别特征列
categorical_features = [
    "season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"
]
# 使用 pd.get_dummies 对类别特征进行 One-Hot 编码，将类别变量转换为多个二元特征
X_encoded = pd.get_dummies(X, columns=categorical_features)
print("d")

# -----------------------------------------------
# 3. 划分训练集和测试集
# -----------------------------------------------
# 这里我们没有使用 sklearn 中的 train_test_split，而是手动划分数据
# 前80%的数据作为训练集，后20%作为测试集
train_ratio = 0.8
X_train = X_encoded[: int(train_ratio * len(dataset))]
X_test = X_encoded[int(train_ratio * len(dataset)) :]
y_train = y[: int(train_ratio * len(dataset))]
y_test = y[int(train_ratio * len(dataset)) :]

# -----------------------------------------------
# 4. 数据标准化
# -----------------------------------------------
# 使用 StandardScaler 对特征进行标准化处理，使得每个特征的均值为0，方差为1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# 在训练集上拟合 scaler 对象，然后用它来转换训练集和测试集
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------
# 5. 训练线性回归模型
# -----------------------------------------------
# 使用 sklearn 中的 LinearRegression 创建一个线性回归模型
from sklearn.linear_model import LinearRegression

linear_reg_model = LinearRegression()
# 使用训练集对模型进行训练，学习输入特征与目标变量之间的关系
linear_reg_model.fit(X_train_scaled, y_train)
# 对测试集进行预测，得到预测值
y_pred = linear_reg_model.predict(X_test_scaled)

# -----------------------------------------------
# 6. 评估模型
# -----------------------------------------------
# 使用均方误差（MSE）来评估模型的预测误差，越小表示模型越好
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
# 计算 R-squared 值，表示模型对目标变量变异的解释程度，值越接近1表示模型拟合得越好
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# -----------------------------------------------
# 7. 可视化结果
# -----------------------------------------------
# 绘制预测值与实际值的散点图，直观地观察模型预测的准确性
import matplotlib.pyplot as plt

plt.figure(0)
# 横坐标为实际值，纵坐标为预测值
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")  # x轴标签：实际值
plt.ylabel("Predicted Values")  # y轴标签：预测值
plt.title("Actual vs Predicted Values in Linear Regression")  # 图标题

# 按照索引对数据进行排序，以便可视化时按顺序展示
sorted_indices = X_test.index.argsort()
y_test_sorted = y_test.iloc[sorted_indices]
# 按排序后的索引排列预测值，方便对比
y_pred_sorted = pd.Series(y_pred).iloc[sorted_indices]

# 绘制排序后的实际值和预测值的曲线，展示模型预测的趋势
plt.figure(1)
plt.plot(y_test_sorted.values, label="Actual Values", marker="o")  # 实际值曲线
plt.plot(y_pred_sorted.values, label="Predicted Values", marker="x")  # 预测值曲线
plt.xlabel("Sample Index (Sorted)")  # x轴标签：样本索引（已排序）
plt.ylabel("Values")  # y轴标签：值
plt.title("Actual vs Predicted Values in Linear Regression")  # 图标题
plt.legend()  # 显示图例
plt.show()  # 展示图形
