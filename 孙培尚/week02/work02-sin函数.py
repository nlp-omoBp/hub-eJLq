import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 修改1：生成sin函数数据（替换原来的线性数据）
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)  # sin函数范围：-2π到2π
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数加上噪声

# 将NumPy数组转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()


# 修改2：构建多层神经网络（替换单层Linear）
class SinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),  # 输入层→隐藏层1（64个神经元）
            nn.ReLU(),  # 激活函数，引入非线性
            nn.Linear(64, 32),  # 隐藏层1→隐藏层2（32个神经元）
            nn.ReLU(),  # 激活函数
            nn.Linear(32, 1)  # 隐藏层2→输出层
        )

    def forward(self, x):
        return self.network(x)


# 创建模型实例
model = SinModel()

# 定义损失函数（均方误差）
loss_fn = nn.MSELoss()

# 修改3：使用Adam优化器（更适合非线性拟合）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 2000  # 增加训练轮次以适应非线性拟合
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化：
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")

# 将模型切换到评估模式
model.eval()

# 禁用梯度计算，进行预测
with torch.no_grad():
    y_predicted = model(X).numpy()

# 修改4：绘图展示sin函数拟合效果
plt.figure(figsize=(12, 6))
# 显示带噪声的训练数据点
plt.scatter(X_numpy, y_numpy, label='带噪声的sin数据', color='blue', alpha=0.6, s=10)
# 显示神经网络拟合的曲线
plt.plot(X_numpy, y_predicted, label='神经网络拟合曲线', color='red', linewidth=2)
# 显示真实的sin函数曲线作为对比
plt.plot(X_numpy, np.sin(X_numpy), label='真实sin函数', color='green', linestyle='--', linewidth=1)

plt.xlabel('X (弧度)')
plt.ylabel('y')
plt.title('多层神经网络拟合Sin函数')
plt.legend()
plt.grid(True)
plt.show()
