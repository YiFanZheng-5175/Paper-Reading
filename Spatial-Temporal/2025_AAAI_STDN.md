# Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting

>领域：时空序列预测  
>发表在：AAAI 2025
>模型名字：***S***patiotemporal-aware ***T***rend-Seasonality ***D***ecomposition ***N***etwork  
>文章链接：[Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting](https://www.arxiv.org/abs/2502.12213)  
>代码仓库：Coming Soon  
![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308231153.png)

## 一、研究背景与问题提出

### 1.1 研究现状

首先，适当的归纳偏差可以更有效地对交通流中突出的时空特征进行建模。尽管先前的研究已经表明，交通模式受到特定时间周期性的强烈影响，但大多数努力只是将这些时间特征大致纳入模型中，而没有***明确地对长周期和短周期之间的协同作用***进行建模。  
关于空间方面，虽然不同的位置表现出独特的空间特征，但大多数图神经网络（GNN）方法仅***基于两个节点之间的距离构建静态图***，未能考虑***图中所有节点之间的全局相互作用***。
其次，对交通流量进行有效的趋势 - 季节性分解可以极大地增强交通节点的表示学习（Autoformer；STWave）。利用这种方法通过区分系统模式和噪声成分来改善交通流量的预测。然而，趋势-季节性分解主要应用于交通网络中的单个节点，***忽略了全局节点之间的相互作用***，从而降低了 GNN 学习到的节点表示的质量。

## 二、问题剖析与解决策略

### 2.1 解决方法

为了弥补这些研究差距，我们引入了一种时空感知趋势季节性分解网络（STDN），该网络通过结合时空嵌入的新颖趋势季节性分解来增强全局节点表示。它具有三个关键模块：（1）时空嵌入学习模块通过学习包含特定周和分钟的时间嵌入来对时间周期性进行建模，并从图拉普拉斯矩阵的特征值和特征向量中获取初始空间位置嵌入。（2）***动态关系图学习模块探索交通节点之间的全局动态交互***，通过时空嵌入进行增强，从而捕获每个交通节点之间的高阶关系。（3）趋势季节性分解模块旨在通过将交通流量分解为趋势周期性和季节性成分来细化节点表示，这些成分进一步通过编码器 - 解码器网络进行处理。

#### 2.1.1 动态关系图学习  

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250309175033.png)  
为了解决基于简单距离的连接性的局限性，这种连接性忽略了每个节点之间的高阶关系，我们构建了一个动态关系图，该图考虑了不同的时间步长和节点。这种方法使我们能够有效地对所有交通节点之间复杂的高阶时空关系进行建模。

#### 2.1.2 趋势-季节性分解

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250309175312.png)

#### 2.1.3 时空趋势季节分解

**分解方法**：在处理交通序列数据时，由于交通隐藏状态和时空嵌入的维度已对齐，通过节点隐藏状态 $H_{L}$ 和时空嵌入 $M$ 的逐元素相乘得到趋势分量，即 $\mathcal{X}_{t}=\mathcal{H}_{L} \odot \mathcal{M}$ ，剩余部分为季节性分量， $\mathcal{X}_{s}=\mathcal{H}_{L}-\mathcal{X}_{t}$ 。这里的趋势分量通常较为平滑，受交通节点的时间和位置影响较大，假设附近时间和位置的趋势相似，不同时间和位置具有独特交通模式。

#### 2.1.4 组合

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250309175707.png)

（评论：在哪里解决了“***明确地对长周期和短周期之间的协同作用***”这个问题）

### 2.2 模型结构

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308231153.png)

## 三、实验验证与结果分析

### 3.1 消融实验

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250309180316.png)

- “w/o TE” 去除时间嵌入建模，让解码器仅依赖空间嵌入线索
- “w/o SE” 去除空间嵌入建模，解码器仅依据时间嵌入线索工作
- “w/o STE” 去除时空嵌入，交通流不分解为趋势季节性分量，解码器不使用任何时空线索；
- “w/o DRG” 去除动态关系图学习
- “w/o STD” 不采用基于时空感知的分解方法，而是使用 Autoformer 的分解方法
