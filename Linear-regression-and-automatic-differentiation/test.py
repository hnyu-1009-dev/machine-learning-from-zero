# 导入库
import tensorflow as tf
import numpy as np
from torchsummary import summary

# 1.散点输入 定义输入数据
data = [
    [-0.5, 7.7],
    [1.8, 98.5],
    [0.9, 57.8],
    [0.4, 39.2],
    [-1.4, -15.7],
    [-1.4, -37.3],
    [-1.8, -49.1],
    [1.5, 75.6],
    [0.4, 34.0],
    [0.8, 62.3],
]
# 转化为数组
data = np.array(data)
# 提取x 和y
x_data = data[:, 0]
y_data = data[:, 1]

# 转化为tensorflow用的张量
x_train = tf.constant(x_data, dtype=tf.float32)
y_train = tf.constant(y_data, dtype=tf.float32)

# 可以将numpy数组或者tensorflow张量
dataset = tf.data.Dataset.from_tensor_slices(
    (
        x_train,
        y_train,
    ),
)
# 使用 tf.data.Dataset.from_tensor_slices 创建数据集
# from_tensor_slices() 返回一个 tf.data.Dataset 对象，这个对象是一个可迭代的数据集，包含了从 tensors 中切分出来的数据。每次从 dataset 中提取数据时，它会返回一个元素（例如：(input, label) 配对）。
dataset = tf.data.Dataset.from_tensor_slices(
    (
        # tensors（输入数据）：可以是一个或多个 NumPy 数组、TensorFlow 张量、Python 列表等，表示模型的输入数据和目标标签。
        # 如果传入多个数组或张量，它们的切片会一起组成数据集中的每个元素。
        # 在此例中，tensors 是一个包含 x_train 和 y_train 的元组。x_train 是输入数据（特征），y_train 是目标标签（输出）。这些数组会按元素拆分并一起组成数据集的元素。
        # 形状要求：
        # 如果 x_train 和 y_train 是 NumPy 数组或 TensorFlow 张量，它们的形状应该一致，确保每个输入数据样本都对应一个目标标签。
        x_train,  # 输入数据，通常是一个 NumPy 数组或 TensorFlow 张量，形状为 (num_samples, num_features)
        y_train,  # 目标数据（标签），通常是一个 NumPy 数组或 TensorFlow 张量，形状为 (num_samples,)
    )
)

# 额外的参数及解释：

# 1. batch_size：指定每个批次中包含多少个样本
#    - 在训练过程中，数据会被拆分成多个批次，每个批次包含 `batch_size` 个样本。
#    - batch_size 越大，内存占用越高，训练可能越稳定，但每次更新权重的频率越低。
#    - 通常 batch_size 设置为 32、64 或 128 等常见值。
dataset = dataset.batch(
    batch_size=32,
    drop_remainder=True,
)
# 解释：
# drop_remainder=True：如果数据集的样本数无法整除批次大小，是否丢弃最后一个批次中的不足样本数。默认为 False。

# 2. shuffle：指定是否在训练时对数据进行打乱
#    - shuffle=True 会在每个 epoch 结束后打乱数据，帮助模型避免对数据顺序产生依赖。
#    - shuffle(buffer_size=10000)：设置缓冲区大小，缓冲区大小越大，打乱效果越好。
dataset = dataset.shuffle(
    buffer_size=10000,
)

# 3. prefetch：预取数据，用于在训练过程中提前加载数据，避免数据加载成为瓶颈
#    - prefetch(buffer_size=tf.data.experimental.AUTOTUNE)：TensorFlow 会自动调整缓冲区大小，确保 CPU 和 GPU 能够并行工作，提高效率。
dataset = dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE,
)

# 4. num_parallel_calls：并行加载数据的数量
#    - num_parallel_calls=tf.data.experimental.AUTOTUNE：让 TensorFlow 自动决定并行处理数据的数量，通常用于加速数据处理。
#    - 如果不指定，数据加载将是串行的，速度可能较慢。
# dataset = dataset.map(map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# 5. drop_remainder：指定是否丢弃最后一个批次中不足 `batch_size` 个样本的数据
#    - drop_remainder=True：如果最后一个批次的样本数不足 `batch_size`，则丢弃这个批次。
#    - drop_remainder=False：保持最后一个批次，即使它的样本数不足 `batch_size`。
#    - 适用于训练和验证数据，保证每个批次的样本数一致。
# dataset = dataset.batch(32, drop_remainder=True)

# 现在可以迭代 dataset，查看每个批次的数据
for batch_x, batch_y in dataset:
    print(batch_x.shape, batch_y.shape)


# tf.keras.Sequential() 是一种模型定义方式，表示按顺序逐层堆叠网络层
# 这里我们创建一个简单的全连接层（Dense层）模型，它有一个输入层和一个输出层
#  (1,) 指的是（None，1）None代表参数的数量不知道有多少
model = tf.keras.Sequential(
    [
        # 定义一层全连接层（Dense layer）
        tf.keras.layers.Dense(
            1,  # 输出维度，表示该层输出一个标量（即只有一个神经元）
            input_shape=(
                1,
            ),  # 输入的形状为 (1,) 表示输入是一个一维的数值（标量），例如一个数值 [x]
        ),
    ]
)
# 定义损失函数和优化器

# 使用 SGD（随机梯度下降）优化器来训练模型
# 随机梯度下降（SGD）是一种常见的优化算法，它通过迭代地调整模型的权重来最小化损失函数
optimizer = tf.keras.optimizers.SGD(
    learning_rate=0.01,  # 学习率：控制每次更新时权重调整的步幅大小。较高的学习率可能导致训练不稳定，较低的学习率可能导致训练过程过慢。
    # 如果不设置学习率，默认值为 0.01
)

# 通过 model.compile() 告诉模型使用哪个优化器和损失函数来进行训练
# 通过编译模型，模型会知道如何根据数据进行训练和权重更新
model.compile(
    optimizer=optimizer,  # 使用上面定义的优化器（SGD）
    loss="mean_squared_error",  # 损失函数：均方误差（MSE），适用于回归任务。它通过计算预测值和真实值之间的差距来评估模型的表现。
    # 损失函数的作用是衡量模型预测的输出和实际标签之间的误差。较小的损失值表示模型预测效果较好。
)
# 定义日志位置
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
#
# #4.开始迭代
epoches = 500
history = model.fit(
    x_train,  # 训练数据的输入部分，形状通常是 (num_samples, num_features)
    y_train,  # 训练数据的目标值（标签），形状通常是 (num_samples,)
    epochs=epoches,  # 训练的轮次，即模型在整个训练数据上训练的次数
    # batch_size=32,       # 每个批次的样本数量，通常选择 32、64 等常见值。如果为 None，则使用整个数据集训练（即每个 epoch 只使用一个批次）
    # validation_data=(x_val, y_val),  # 验证数据集，使用这个数据集来评估模型在每个 epoch 结束时的表现
    # validation_split=0.2,             # 在训练数据中划分出 20% 的数据作为验证集，剩余的用于训练
    verbose=0,  # 控制训练过程中的日志显示：0 = 不显示进度，1 = 显示进度条，2 = 每个 epoch 输出一行日志
    # callbacks=[early_stopping],  # 训练过程中使用的回调函数，例如 EarlyStopping 用于避免过拟合
    # shuffle=True            # 是否在每个 epoch 后打乱数据，通常设置为 True
    # initial_epoch=0         # 从哪个 epoch 开始训练，通常用于恢复训练
    callbacks=[tensorboard_callback],  # 生成tensorboard所用到的日志
)

# 1. 保存整个模型（包括结构、权重和训练配置） -> HDF5 格式
# 保存完整模型，包括网络结构、权重和训练配置（优化器、损失函数等）。
# 适用于跨平台保存和加载。文件扩展名为 .h5
model.save("./model/save_model.h5")
# 解释：
# model.save() 会将整个模型（结构 + 权重 + 配置）保存为一个文件，可以直接用 `model.load()` 重新加载模型。

# 2. 只保存模型的权重（不保存模型结构） -> HDF5 格式
# 这种方式只保存模型的权重，包括模型的参数（权重 w 和偏置 b）。
# 适用于你已经有模型结构代码，且只需要恢复训练过程中的权重。
model.save_weights("./model/model.weights.h5")
# 解释：
# model.save_weights() 只保存权重（没有保存模型的结构）。如果你想在其他地方加载权重，需要重新定义相同的模型结构。

# 3. 使用 TensorFlow 保存模型（推荐使用 .keras 扩展名）
# 适用于 TensorFlow 环境，可以使用 `.keras` 格式保存模型。此格式更适合在 TensorFlow 环境中进行部署（如 TensorFlow Serving）。
model.save("./model/save_model.keras")
# 解释：
# `.keras` 是 TensorFlow 推荐的保存格式，能够将整个模型（包括权重、结构和配置）保存到文件中。可以在不同的 TensorFlow 环境中加载。

# 4. 使用 TensorFlow 格式保存模型（使用 .h5 扩展名） -> HDF5 格式
# 如果你习惯使用 `.h5` 格式保存模型，这也是常见的选择，适合与旧的 TensorFlow 版本兼容。
model.save("./model/save_model.h5")
# 解释：
# `.h5` 格式仍然被广泛使用，特别是在 TensorFlow 2.x 版本之前。它保存了整个模型（包括结构和权重），适合跨平台部署。

# 5. 保存模型为 SavedModel 格式（TensorFlow 专有格式）
# SavedModel 格式适用于 TensorFlow 和 TensorFlow Serving，是 TensorFlow 官方推荐的模型保存格式。
# 适合生产环境部署，支持跨语言和跨平台的使用。
# save_format 参数已被弃用
# model.save("./saved_model", save_format="tf")
# 解释：
# `save_format="tf"` 指定使用 TensorFlow 原生格式（SavedModel）保存模型。SavedModel 格式会将模型保存到文件夹中，包括模型结构、权重和优化器配置等。

# 默认是GPU如果是cpu最好指定一下
# 如果安装了gpu版的pytorch 如果不指定可能会报错
# summary() 的常见输出包括：
# Layer (type)：显示每个网络层的类型（例如 Linear、Conv2d 等）。
# Output Shape：显示每个层的输出形状（例如 (10, 1)）。  如果是-1则表示该维度大小不固定
# ！！！！！这里一定要搞清楚Output Shape说明的是输出tesner的形状
# nn.Linear(1, 1)表示的是输入特征是1 输出特征也是1
# Param #：显示每个层的参数数量。
# Total params：显示模型的总参数数量。
model.summary()