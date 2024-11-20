import numpy as np
import matplotlib.pyplot as plt

# 定义数据集
x_data = [1.0, 2.0, 3.0]
y_data = [1.9, 3.9, 5.9]

# 生成矩阵坐标
W, B = np.arange(0.0, 4.1, 0.1).round(1), np.arange(-2.0, 2.1, 0.1).round(1)
w, b = np.meshgrid(W, B)

# 定义模型：y = x * w - b
def forward(x):
	return x * w + b

# 损失函数
def loss(y_pre, y):
	return (y_pre - y) ** 2

l_sum = 0 # 计算损失之和
for x_val, y_val in zip(x_data, y_data):
	y_pre = forward(x_val)
	loss_val = loss(y_pre, y_val)
	l_sum += loss_val
mse = l_sum / len(x_data)

# 定义figure
fig = plt.figure(figsize=(2.5, 2.5), dpi=210)

# 画3D图
ax = plt.axes(projection='3d')
surf = ax.plot_surface(w, b, mse, rstride=1, cstride=1, cmap='rainbow')

# 设置z轴
ax.set_zlim(0, 40)

# 设置图标
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.text(0.2, 2, 43, 'Cost Value', color='black')

# 增加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
