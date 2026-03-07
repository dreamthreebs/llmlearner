# 术语表（Glossary）

| 缩写/术语 | 全称 | 简述 |
|-----------|------|------|
| Adam | Adaptive Moment Estimation | 自适应学习率优化器，维护梯度的一阶矩（均值）和二阶矩（未中心方差）的指数移动平均 |
| AdaGrad | Adaptive Gradient Algorithm | 对每个参数根据历史梯度平方和自适应调整学习率的优化器 |
| Adjacency Matrix | 邻接矩阵 | 用矩阵表示图中节点之间连接关系的数据结构 |
| AlexNet | — | 2012年ImageNet冠军模型，首次大规模使用ReLU和Dropout的深度CNN |
| Autoencoder | 自编码器 | 将输入压缩为低维潜在表示再重建输出的无监督神经网络 |
| Autograd | 自动微分 | PyTorch中基于计算图自动计算梯度的引擎 |
| Backpropagation | 反向传播 | 利用链式法则从输出层到输入层逐层计算梯度的算法 |
| Batch Normalization (BN) | 批量归一化 | 在每个mini-batch内对激活值做归一化，加速训练并稳定梯度 |
| Bias | 偏置 | 神经元中独立于输入的可学习常数项，控制激活函数的平移 |
| Bias-Variance Tradeoff | 偏差-方差权衡 | 模型复杂度增加时偏差下降但方差上升的泛化误差分解框架 |
| BPTT | Backpropagation Through Time | 将RNN沿时间步展开后应用反向传播的训练算法 |
| Cell State | 细胞状态 | LSTM中贯穿整个序列的长期记忆通道，通过门机制进行读写 |
| CNN | Convolutional Neural Network | 利用卷积操作提取空间局部特征的神经网络，广泛用于图像任务 |
| Convolution | 卷积 | 用可学习的滤波器在输入上滑动做局部加权求和的操作 |
| Cross-Entropy Loss | 交叉熵损失 | 衡量预测概率分布与真实标签分布之间差异的分类损失函数 |
| DataLoader | 数据加载器 | PyTorch中负责批量加载、打乱和预处理数据的工具类 |
| DCGAN | Deep Convolutional GAN | 用卷积/反卷积替代全连接层的GAN架构，生成更稳定的图像 |
| Decoder | 解码器 | 将潜在表示映射回原始数据空间的网络部分 |
| Deep Learning | 深度学习 | 使用多层神经网络从数据中自动学习层次化表示的机器学习方法 |
| Diffusion Model | 扩散模型 | 通过逐步加噪再学习去噪过程来生成数据的生成模型 |
| Discriminator | 判别器 | GAN中负责判断输入数据是真实的还是生成的网络 |
| Double Descent | 双重下降 | 测试误差随模型复杂度先降后升再降的非经典泛化现象 |
| Dropout | — | 训练时随机将部分神经元输出置零的正则化技术 |
| ELBO | Evidence Lower Bound | 变分推断中对数据对数似然的下界，VAE的优化目标 |
| Embedding | 嵌入 | 将离散对象（如词、节点）映射为连续向量表示的方法 |
| Encoder | 编码器 | 将输入数据压缩为低维潜在表示的网络部分 |
| Epoch | 训练轮次 | 整个训练数据集被完整遍历一次的过程 |
| Feature Map | 特征图 | 卷积层输出的二维激活矩阵，表示检测到的局部特征 |
| FFN | Feed-Forward Network | 信息仅从输入到输出单向流动、无循环连接的神经网络 |
| Fine-tuning | 微调 | 在预训练模型基础上用特定任务数据继续训练的迁移学习策略 |
| Forget Gate | 遗忘门 | LSTM中控制细胞状态中哪些旧信息应被遗忘的门机制 |
| GAN | Generative Adversarial Network | 通过生成器与判别器对抗训练来生成逼真数据的框架 |
| GAT | Graph Attention Network | 使用注意力机制对邻居节点加权聚合的图神经网络 |
| GCN | Graph Convolutional Network | 将卷积操作推广到图结构数据上的图神经网络 |
| GELU | Gaussian Error Linear Unit | 基于高斯累积分布函数的平滑激活函数，常用于Transformer |
| Generator | 生成器 | GAN中从随机噪声生成逼真数据的网络 |
| GNN | Graph Neural Network | 在图结构数据上通过消息传递学习节点/图表示的神经网络 |
| Gradient Clipping | 梯度裁剪 | 当梯度范数超过阈值时进行缩放，防止梯度爆炸的技术 |
| Gradient Explosion | 梯度爆炸 | 深层网络中梯度在反向传播过程中指数级增大导致训练不稳定的现象 |
| Gradient Vanishing | 梯度消失 | 深层网络中梯度在反向传播过程中指数级减小导致浅层无法更新的现象 |
| Graph | 图 | 由节点和边组成的数据结构，用于表示实体间的关系 |
| GRU | Gated Recurrent Unit | LSTM的简化变体，合并遗忘门和输入门为更新门，参数更少 |
| He Initialization | He初始化 | 针对ReLU激活函数设计的权重初始化方法，方差为2/n |
| Hidden State | 隐状态 | RNN中在时间步之间传递的内部记忆向量 |
| Hyperparameter | 超参数 | 训练前需人工设定、不由模型自动学习的参数（如学习率、层数） |
| Implicit Regularization | 隐式正则化 | 优化算法（如SGD）在训练过程中隐含的偏向简单解的正则化效果 |
| Input Gate | 输入门 | LSTM中控制当前输入的哪些信息应被写入细胞状态的门机制 |
| Kernel | 卷积核/滤波器 | 卷积操作中用于特征提取的可学习权重矩阵 |
| KL Divergence | Kullback-Leibler Divergence | 衡量两个概率分布差异的非对称度量 |
| L1 Regularization | L1正则化 | 在损失函数中加入权重绝对值之和的惩罚项，倾向于产生稀疏解 |
| L2 Regularization | L2正则化（权重衰减） | 在损失函数中加入权重平方和的惩罚项，防止权重过大 |
| Latent Representation | 潜在表示 | 数据被编码器压缩后的低维内部表示 |
| Latent Space | 潜在空间 | 潜在表示所在的低维连续空间 |
| Layer Normalization (LN) | 层归一化 | 对单个样本的所有特征做归一化，不依赖batch大小 |
| Learning Rate | 学习率 | 控制每次参数更新步长大小的超参数 |
| Learning Rate Schedule | 学习率调度 | 训练过程中按策略动态调整学习率的方法（如余弦退火、warmup） |
| LeNet | — | LeCun提出的早期CNN架构，用于手写数字识别 |
| Linear Layer | 线性层/全连接层 | 对输入做仿射变换 y=Wx+b 的网络层 |
| Loss Function | 损失函数 | 量化模型预测与真实值之间差异的可微函数 |
| Loss Surface | 损失面 | 损失函数关于模型参数构成的高维曲面 |
| LSTM | Long Short-Term Memory | 通过门机制解决长程依赖问题的循环神经网络变体 |
| Max Pooling | 最大池化 | 取局部区域中最大值的下采样操作，保留最显著特征 |
| Mechanistic Interpretability | 机制性可解释性 | 通过逆向工程分析神经网络内部计算机制的研究方向 |
| Message Passing | 消息传递 | GNN中节点从邻居收集信息并更新自身表示的核心操作框架 |
| Mini-batch | 小批量 | 每次参数更新使用的训练数据子集 |
| Minimax | 极小极大博弈 | GAN的优化目标：生成器最小化、判别器最大化同一目标函数 |
| MLP | Multi-Layer Perceptron | 由多个全连接层和激活函数组成的前馈神经网络 |
| Mode Collapse | 模式坍塌 | GAN训练中生成器只产生少数几种样本、缺乏多样性的失败模式 |
| Momentum | 动量 | 在梯度更新中累积历史梯度方向的加速技术，帮助越过局部极小 |
| MSE Loss | Mean Squared Error Loss | 预测值与真实值差的平方的均值，常用于回归任务的损失函数 |
| nn.Module | — | PyTorch中所有神经网络模块的基类，管理参数和前向计算 |
| NTK | Neural Tangent Kernel | 描述无限宽网络训练动态等价于核方法的理论框架 |
| Output Gate | 输出门 | LSTM中控制细胞状态的哪些信息输出为当前隐状态的门机制 |
| Over-smoothing | 过平滑 | GNN层数增多时节点表示趋于一致、丧失区分能力的现象 |
| Overfitting | 过拟合 | 模型在训练集上表现好但在测试集上泛化差的现象 |
| PAC Learning | Probably Approximately Correct | 在概率框架下分析学习算法样本复杂度的计算学习理论 |
| Padding | 填充 | 在输入边缘补零以控制卷积输出尺寸的操作 |
| PCA | Principal Component Analysis | 通过正交变换将数据投影到最大方差方向的线性降维方法 |
| Perceptron | 感知机 | 最简单的单层线性分类神经网络模型 |
| Pooling | 池化 | 对特征图进行空间下采样以降低维度和增强平移不变性的操作 |
| Rademacher Complexity | Rademacher复杂度 | 衡量假设空间拟合随机噪声能力的泛化理论复杂度度量 |
| Receptive Field | 感受野 | 网络某层神经元在输入空间中能"看到"的区域大小 |
| ReLU | Rectified Linear Unit | f(x)=max(0,x)，最常用的激活函数，缓解梯度消失 |
| Reparameterization Trick | 重参数化技巧 | 将随机采样过程改写为确定性函数加噪声，使梯度可以回传 |
| Residual Connection | 残差连接/跳跃连接 | 将层的输入直接加到输出上，缓解深层网络的梯度消失问题 |
| ResNet | Residual Network | 引入残差连接使训练极深网络成为可能的CNN架构 |
| RNN | Recurrent Neural Network | 具有循环连接、能处理序列数据的神经网络 |
| Saddle Point | 鞍点 | 损失面上某些方向是极小、某些方向是极大的临界点 |
| Scaling Laws | 缩放定律 | 描述模型性能随参数量、数据量和计算量幂律增长关系的经验规律 |
| SGD | Stochastic Gradient Descent | 每次用随机小批量样本估计梯度进行参数更新的优化算法 |
| Sigmoid | Sigmoid函数 | 将输入压缩到(0,1)区间的S型激活函数 σ(x)=1/(1+e^(-x)) |
| Softmax | — | 将向量转化为概率分布的函数，常用于多分类输出层 |
| Sparse Autoencoder | 稀疏自编码器 | 通过稀疏约束学习过完备但稀疏的特征表示的自编码器变体 |
| Stride | 步幅 | 卷积核或池化窗口每次滑动的像素距离 |
| StyleGAN | Style-based GAN | 通过风格向量逐层控制生成图像属性的高质量图像生成架构 |
| Tanh | 双曲正切函数 | 将输入压缩到(-1,1)区间的激活函数 |
| Tensor | 张量 | 多维数组的统称，PyTorch中基本的数据结构和计算单元 |
| Underfitting | 欠拟合 | 模型复杂度不足，无法捕捉数据中规律的现象 |
| VAE | Variational Autoencoder | 通过变分推断在潜在空间中学习数据分布的生成模型 |
| VC Dimension | Vapnik-Chervonenkis维 | 衡量假设空间复杂度的指标，等于能被该空间打散的最大点集大小 |
| VGGNet | — | 使用堆叠3×3小卷积核构建深层网络的CNN架构 |
| Warmup | 学习率预热 | 训练初期从小学习率逐步增大到目标值的调度策略 |
| Wasserstein Distance | Wasserstein距离（推土机距离） | 衡量两个概率分布差异的度量，WGAN的核心改进 |
| Weight Decay | 权重衰减 | 等价于L2正则化，在每次更新时对权重乘以衰减系数 |
| Weight Initialization | 权重初始化 | 训练开始前为网络权重赋初值的策略，影响训练稳定性和收敛速度 |
| WGAN | Wasserstein GAN | 使用Wasserstein距离替代JS散度作为训练目标的GAN变体，训练更稳定 |
| Xavier Initialization | Xavier初始化 | 针对Sigmoid/Tanh设计的权重初始化方法，保持各层方差一致 |
