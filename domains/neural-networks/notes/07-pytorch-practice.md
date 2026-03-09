# 07 - PyTorch 实战：从零手写神经网络

> **主维度**：D4 工程实现
> **关键关系**：PyTorch `used-for` 实现所有 D1 架构和 D2 范式
>
> **学习路径**：Step 6 / 7  
> **前置知识**：MLP、CNN 架构原理（01-overview）、损失函数与反向传播、正则化（06-generalization）  
> **参考**：  
> - [PyTorch 官方教程](https://pytorch.org/tutorials/)  
> - [Dive into Deep Learning (d2l.ai)](https://d2l.ai/)  
> - [Karpathy - Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)  
> - [PyTorch 官方文档](https://pytorch.org/docs/stable/)

---

## 核心问题

前面的章节里我们学了 MLP、CNN、RNN 的架构原理，学了损失函数、反向传播、优化器的数学，也学了正则化和泛化理论。现在的问题是：**怎么把这些理论变成实际可运行的代码？**

本章用 Python + PyTorch 从零实现两个完整项目：
1. 用 MLP 做 MNIST 手写数字分类
2. 用 CNN 做 CIFAR-10 图像分类

每一步都先解释概念，再给代码。

---

## 1. PyTorch 核心概念

### 1.1 Tensor（张量）

**Tensor**（张量）是 PyTorch 中最基本的数据结构。如果你用过 NumPy，可以把 tensor 理解为"能在 GPU 上运算的 ndarray"。和 NumPy 的 ndarray 一样，tensor 是一个多维数组；不同的是，tensor 可以自动追踪所有对它的运算，从而支持自动求导（下一节会讲）。

```python
import torch

# 创建张量的几种方式
x = torch.tensor([1.0, 2.0, 3.0])            # 从列表创建
x = torch.zeros(3, 4)                         # 3×4 全零矩阵
x = torch.randn(3, 4)                         # 3×4 标准正态随机矩阵
x = torch.arange(0, 10, 2)                    # [0, 2, 4, 6, 8]

# 基本属性
print(x.shape)                                 # 形状
print(x.dtype)                                 # 数据类型（默认 float32）
print(x.device)                                # 所在设备（cpu 或 cuda）

# 基本运算（和 NumPy 几乎一样）
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = a + b                                     # 逐元素加法
d = a @ b.T                                   # 矩阵乘法（3×4 乘 4×3 → 3×3）
e = a * b                                     # 逐元素乘法（Hadamard 积）

# 和 NumPy 互转
import numpy as np
np_array = x.numpy()                           # tensor → numpy（共享内存）
tensor = torch.from_numpy(np_array)            # numpy → tensor

# GPU 操作（如果有 NVIDIA GPU）
if torch.cuda.is_available():
    x = x.to('cuda')                          # 移到 GPU
    x = x.to('cpu')                           # 移回 CPU
```

> 参考：[PyTorch Tensor 文档](https://pytorch.org/docs/stable/tensors.html)

### 1.2 自动微分（Autograd）

**自动微分**（automatic differentiation，简称 autograd）是 PyTorch 的核心功能。你在前面的章节中学过反向传播——本质上就是链式法则的自动化应用。PyTorch 的 autograd 系统能自动追踪你对 tensor 做的所有运算，构建一个 **计算图**（computational graph），然后沿着这个图反向计算梯度。

一个最简单的例子——计算 $y = x^2$ 在 $x = 3$ 处的导数 $\frac{dy}{dx} = 2x = 6$：

```python
import torch

x = torch.tensor(3.0, requires_grad=True)  # requires_grad=True 告诉 PyTorch 追踪 x 的运算
y = x ** 2                                  # y = x^2 = 9
y.backward()                                # 反向传播，计算 dy/dx
print(x.grad)                               # 输出：tensor(6.)，因为 dy/dx = 2x = 6
```

一个稍复杂的例子——$z = (x + y)^2$，计算 $\frac{\partial z}{\partial x}$ 和 $\frac{\partial z}{\partial y}$：

```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = (x + y) ** 2                            # z = (2+3)^2 = 25

z.backward()
print(x.grad)                               # 10.0，因为 dz/dx = 2(x+y) = 10
print(y.grad)                               # 10.0，因为 dz/dy = 2(x+y) = 10
```

在训练神经网络时，autograd 自动完成了反向传播——你只需要写前向计算，PyTorch 帮你算所有参数的梯度。

> 参考：[PyTorch Autograd 教程](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

### 1.3 nn.Module（定义网络的标准方式）

**`nn.Module`** 是 PyTorch 中定义神经网络的基类。所有自定义网络都继承它。你需要做两件事：

1. **`__init__`**：定义网络的层（有哪些参数）
2. **`forward`**：定义前向传播（数据怎么流过这些层）

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)    # 全连接层：784 个输入 → 256 个输出
        self.fc2 = nn.Linear(256, 10)     # 全连接层：256 个输入 → 10 个输出
        self.relu = nn.ReLU()             # ReLU 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))        # 第一层 + 激活
        x = self.fc2(x)                   # 第二层（输出层不加激活，因为后面会用 CrossEntropyLoss）
        return x

model = SimpleNet()
print(model)                              # 打印网络结构
```

`nn.Linear(in_features, out_features)` 就是一个全连接层，它做的运算是 $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$，其中 $\mathbf{W}$ 是 `out_features × in_features` 的权重矩阵，$\mathbf{b}$ 是偏置向量。这些参数由 PyTorch 自动管理，你不需要手动创建。

> 参考：[PyTorch nn.Module 文档](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

### 1.4 DataLoader（数据加载）

**`DataLoader`** 是 PyTorch 的数据加载器，它负责把数据集分成小批量（mini-batch），并在训练时自动打乱顺序。

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),                  # 把 PIL 图像转为 tensor，值从 [0,255] 缩放到 [0,1]
    transforms.Normalize((0.5,), (0.5,))    # 标准化到 [-1, 1]
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 遍历数据
for images, labels in train_loader:
    print(images.shape)                     # torch.Size([64, 1, 28, 28])：64张图，1通道，28×28
    print(labels.shape)                     # torch.Size([64])：64个标签
    break
```

### 1.5 优化器（Optimizer）

**优化器**负责根据梯度更新模型参数。你在前面学过 SGD 和 Adam 的数学——PyTorch 把这些封装好了，你只需要告诉它"更新哪些参数"和"学习率是多少"。

```python
import torch.optim as optim

model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 一个典型的训练步骤：
optimizer.zero_grad()                       # 清除上一步的梯度（PyTorch 默认累加梯度）
output = model(input_data)                  # 前向传播
loss = criterion(output, target)            # 计算损失
loss.backward()                             # 反向传播，计算梯度
optimizer.step()                            # 用梯度更新参数
```

注意 `optimizer.zero_grad()` 这一步：PyTorch 默认把梯度 **累加** 而不是替换（这在某些高级用法中有用），所以每次反向传播前必须手动清零。

---

## 2. 实战 1：MLP 做 MNIST 手写数字分类

MNIST 是机器学习的"Hello World"数据集：28×28 像素的手写数字灰度图像，共 10 类（0-9），60000 张训练图 + 10000 张测试图。

### 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ========================
# 1. 超参数
# ========================
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# 2. 数据加载
# ========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST 的全局均值和标准差
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================
# 3. 定义 MLP 网络
# ========================
class MLP(nn.Module):
    """
    三层全连接网络：784 → 256 → 128 → 10
    输入：28×28 = 784 维的展平图像
    输出：10 维向量（每个类别的得分）
    """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()           # 把 28×28 展平为 784
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),                  # 20% 的 dropout 防止过拟合
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)                # 输出层不加激活（CrossEntropyLoss 内部会做 softmax）
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

model = MLP().to(DEVICE)
print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
# 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = 235,146

# ========================
# 4. 损失函数和优化器
# ========================
criterion = nn.CrossEntropyLoss()             # 交叉熵损失（分类任务的标准选择）
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ========================
# 5. 训练循环
# ========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()                             # 切换到训练模式（启用 dropout）
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                 # 清除旧梯度
        outputs = model(images)               # 前向传播
        loss = criterion(outputs, labels)     # 计算损失
        loss.backward()                       # 反向传播
        optimizer.step()                      # 更新参数

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)         # 取概率最大的类别
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

# ========================
# 6. 测试评估
# ========================
def evaluate(model, loader, criterion, device):
    model.eval()                              # 切换到评估模式（关闭 dropout）
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():                     # 不需要计算梯度（节省内存和计算）
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

# ========================
# 7. 开始训练
# ========================
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# 预期输出（大约）：
# Epoch 10/10 | Train Loss: 0.05, Train Acc: 0.9850 | Test Loss: 0.07, Test Acc: 0.9770
```

### 代码要点解读

**训练循环的五个步骤**——每一个 mini-batch 都重复这个模式：

1. `optimizer.zero_grad()` — 清零梯度
2. `outputs = model(images)` — 前向传播
3. `loss = criterion(outputs, labels)` — 计算损失
4. `loss.backward()` — 反向传播（计算所有参数的梯度）
5. `optimizer.step()` — 用梯度更新参数

这五步是 PyTorch 训练的固定模板，几乎所有项目都遵循这个结构。

**`model.train()` vs `model.eval()`**：某些层（如 Dropout、BatchNorm）在训练和测试时的行为不同。`model.train()` 启用训练行为（如 dropout 随机丢弃神经元），`model.eval()` 启用评估行为（如 dropout 不再丢弃）。忘记切换是一个非常常见的 bug。

**`torch.no_grad()`**：在评估时不需要计算梯度，用这个上下文管理器可以禁用梯度计算，节省内存和加速。

---

## 3. 实战 2：CNN 做 CIFAR-10 图像分类

CIFAR-10 比 MNIST 复杂得多：32×32 像素的 **彩色** 图像（3 通道 RGB），共 10 类（飞机、汽车、鸟、猫、鹿、狗、蛙、马、船、卡车），50000 张训练图 + 10000 张测试图。

MLP 在 CIFAR-10 上效果不好（约 55%），因为 MLP 把图像展平为一维向量后丢失了空间结构。CNN 通过卷积核保留了空间信息。

### 完整代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ========================
# 1. 超参数
# ========================
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================
# 2. 数据加载（含数据增强）
# ========================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),        # 随机水平翻转（数据增强）
    transforms.RandomCrop(32, padding=4),     # 随机裁剪：先填充 4 像素再随机裁回 32×32
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),   # CIFAR-10 的 RGB 通道均值
                         (0.2470, 0.2435, 0.2616))    # CIFAR-10 的 RGB 通道标准差
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ========================
# 3. 定义 CNN 网络
# ========================
class CNN(nn.Module):
    """
    卷积网络结构：
    Conv(3→32) → Conv(32→64) → Pool → Conv(64→128) → Conv(128→128) → Pool → FC(2048→256) → FC(256→10)

    每个 Conv 层后跟 BatchNorm 和 ReLU。
    BatchNorm 对每个通道做归一化，加速训练并提供轻微正则化。
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 第一个卷积块：输入 3 通道（RGB），输出 32 个特征图
            nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 3×32×32 → 32×32×32
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32×32×32 → 64×32×32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 64×32×32 → 64×16×16

            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 64×16×16 → 128×16×16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 第四个卷积块
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# 128×16×16 → 128×16×16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 128×16×16 → 128×8×8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # 128×8×8 = 8192 → 展平
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN().to(DEVICE)
print(f"CNN 参数量：{sum(p.numel() for p in model.parameters()):,}")

# ========================
# 4. 损失函数和优化器
# ========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # weight_decay 即 L2 正则化

# ========================
# 5. 训练和评估（复用前面定义的函数）
# ========================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


# ========================
# 6. 开始训练
# ========================
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# 预期输出（大约）：
# Epoch 20/20 | Train Loss: 0.15, Train Acc: 0.9500 | Test Loss: 0.45, Test Acc: 0.8700
```

### MLP vs CNN 对比

| 指标 | MLP (MNIST) | CNN (CIFAR-10) |
|------|-------------|----------------|
| 数据复杂度 | 28×28 灰度 | 32×32 彩色（难得多） |
| 参数量 | ~235K | ~380K |
| 测试准确率 | ~97% | ~87% |
| 核心优势 | 简单 | 利用空间结构 |

如果用 MLP 做 CIFAR-10，准确率只有约 55%——因为 MLP 把图像展平后，一个像素和它相邻像素的关系、和远处像素的关系被同等对待了。CNN 的卷积核只看局部区域，天然利用了图像的空间局部性。

---

## 4. 训练技巧

### 4.1 学习率调度（Learning Rate Scheduling）

固定学习率通常不是最优的：训练初期需要大学习率快速收敛，后期需要小学习率精细调整。**余弦退火**（Cosine Annealing）是最常用的学习率调度策略之一，学习率按余弦曲线从初始值衰减到接近零：

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

其中 $\eta_t$ 是第 $t$ 步的学习率，$T$ 是总步数。

```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    scheduler.step()                          # 每个 epoch 结束后更新学习率
    print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}")
```

### 4.2 梯度裁剪（Gradient Clipping）

当梯度过大时（通常发生在 RNN 或训练不稳定时），参数更新会"爆炸"。**梯度裁剪** 对梯度向量的范数设置一个上限：如果梯度范数超过阈值 `max_norm`，就按比例缩小所有梯度，使范数恰好等于阈值。

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 4.3 保存和加载模型

训练好的模型需要保存下来，以便后续使用或继续训练。

```python
# 保存模型参数（推荐方式）
torch.save(model.state_dict(), 'model_weights.pth')

# 加载模型参数
model = CNN()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 保存完整 checkpoint（包含优化器状态，用于恢复训练）
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# 从 checkpoint 恢复训练
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### 4.4 用 tqdm 显示训练进度

`tqdm` 是一个进度条库，让训练过程有可视化的进度反馈。上面的代码已经用了 `tqdm` 包裹 DataLoader。安装方式：`pip install tqdm`。

```python
from tqdm import tqdm

for images, labels in tqdm(train_loader, desc="Training"):
    # ... 训练代码 ...
    pass
# 输出：Training: 100%|██████████| 469/469 [00:05<00:00, 85.23it/s]
```

---

## 5. 常见 Bug 和调试技巧

### 5.1 维度不匹配（最常见的错误）

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x8192 and 2048x256)
```

这意味着全连接层期望的输入大小和实际数据不匹配。调试方法——在 `forward` 中打印中间张量的形状：

```python
def forward(self, x):
    x = self.features(x)
    print(f"features 输出形状: {x.shape}")   # 调试用，找到正确的展平后维度
    x = self.classifier(x)
    return x
```

一种更省事的做法是使用 `nn.LazyLinear`，它会在第一次前向传播时自动推断输入大小：

```python
self.fc = nn.LazyLinear(256)    # 输入大小自动推断，输出 256
```

### 5.2 忘记 model.train() / model.eval()

如果训练时忘记调用 `model.train()`，Dropout 层不工作、BatchNorm 使用的是运行均值。如果评估时忘记调用 `model.eval()`，Dropout 仍然在随机丢弃神经元，测试结果每次都不同且偏低。

**规则**：训练循环开头总是 `model.train()`，评估循环开头总是 `model.eval()`。

### 5.3 学习率太大导致 loss 爆炸

如果训练开始后 loss 不降反升、甚至变成 `inf` 或 `nan`，第一个要检查的就是学习率。

```
Epoch 1 | Loss: 2.3026
Epoch 2 | Loss: 15.442
Epoch 3 | Loss: nan
```

解决方法：
- 把学习率降低一个数量级（如 0.001 → 0.0001）
- 使用学习率预热（warmup）：前几个 epoch 用很小的学习率，再逐渐增大

```python
# 简单的 warmup 实现
warmup_epochs = 5
for epoch in range(EPOCHS):
    if epoch < warmup_epochs:
        lr = LEARNING_RATE * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
```

### 5.4 梯度为 NaN

`NaN` 梯度通常由以下原因引起：
- **数值溢出**：中间计算结果太大（如 `exp(1000)`）
- **除以零**：比如自定义损失函数中有除法，分母可能为零
- **学习率太大**：参数更新幅度太大，导致下一步的损失计算溢出

调试方法——检查梯度中是否有 NaN：

```python
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")
```

使用 `torch.autograd.set_detect_anomaly(True)` 可以让 PyTorch 在检测到 NaN 时报告具体是哪一步运算出了问题：

```python
with torch.autograd.detect_anomaly():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
```

---

## 本章总结

| 概念 | 作用 |
|------|-----|
| Tensor | PyTorch 的基本数据结构，支持 GPU 计算和自动微分 |
| Autograd | 自动构建计算图、反向传播求梯度 |
| nn.Module | 定义网络结构的基类（`__init__` + `forward`） |
| DataLoader | 分批加载数据、自动打乱 |
| train/eval 模式 | 控制 Dropout/BatchNorm 的行为 |
| 训练循环模板 | zero_grad → forward → loss → backward → step |

关键心得：
- PyTorch 的设计哲学是 **"define-by-run"**（动态图）——你写 Python 代码就是在定义计算图，调试和普通 Python 一样方便
- 训练循环的五步模板是不变的核心，所有项目都是在这个框架上加功能
- 大部分 bug 都是维度不匹配和 train/eval 模式忘记切换

---

## 理解检测

**Q1**：在训练循环中，`optimizer.zero_grad()` 如果放在 `loss.backward()` 之后、`optimizer.step()` 之前，会发生什么？训练还能正确进行吗？（提示：想想 PyTorch 梯度累加的默认行为。）

你的回答：



**Q2**：假设你的 CNN 在 CIFAR-10 上训练准确率 99% 但测试准确率只有 70%，明显过拟合了。请列出至少 3 种你会尝试的解决方法，并说明每种方法的原理（可以参考 06-generalization.md 中的内容）。

你的回答：



**Q3**：在 CNN 代码中，`nn.Conv2d(3, 32, kernel_size=3, padding=1)` 这一层有多少个可训练参数？请计算（提示：包括权重和偏置，权重是 `out_channels × in_channels × kernel_h × kernel_w`）。

你的回答：


