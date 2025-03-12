# MegaCRN: Meta-Graph Convolutional Recurrent Network for Spatio-Temporal Modeling

>领域：时空序列预测  
>发表在：AAAI 2023  
>模型名字：  ***Me***ta-***G***r***a***ph ***C***onvolutional ***R***ecurrent ***N***etwork  
>文章链接：[MegaCRN: Meta-Graph Convolutional Recurrent Network for Spatio-Temporal Modeling](https://arxiv.org/abs/2212.05989)  
>代码仓库：[https://github.com/deepkashiwa20/megacrn](https://github.com/deepkashiwa20/megacrn)
![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306200911.png)

## 一、研究背景与问题提出

### 1. 1 研究现状

随着图卷积网络（GCNs）的发展，相关研究取得显著进展。主流模型通过利用基于 GCN 的模块和序列模型，如循环神经网络（RNNs）、WaveNet、Transformer 等，对传感器间的潜在空间相关性和时间序列内的时间自相关性进行建模。在图结构学习方面，早期方法依赖路网自然拓扑或根据特定度量（如欧氏距离、逆距离高斯核、余弦相似性）及经验法则预定义图 。后来，GWNet ***首先***提出将邻接矩阵作为自由变量进行训练，生成自适应图，MTGNN、AGCRN 等模型进一步改进了自适应图学习方法；还有模型如 StemGNN 利用自注意力机制学习潜在图结构。

### 1.2 现存问题

虽然一些研究尝试通过矩阵或张量分解、注意力机制来处理网络动态性，但时空异质性和非平稳性仍未得到妥善解决。现有方法***无法很好地区分不同性质的传感器信号***，也***未对交通事件进行有效处理***。

### 1.3 引出思考

能否更好的区分不同性质的传感器信号？
能否对交通事件进行有效处理？

## 二、问题剖析与解决策略

### 2.1 解决方法

#### 2.2.1 时空Meta Graph学习

因此，我们提出了一种新颖的时空元图学习框架。具体来说，我们的 STG 学习包括两个步骤：（1）从元节点库中查询节点级原型；（2）使用超网络重建节点嵌入。这种局部记忆能力使我们的模块化元图学习器***能够在本质上区分不同道路上随时间变化的交通模式***，甚至可以***推广到事件情况***。

![2](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201013.png)

***Comment***：Node emb是静态的，没办法根据输入的不一样来改变，那就通过Encoder的输出（包含了历史信息）来从Node emb中 query出输出，然后这个输出在一起结合放到Decoder里生成信息。

## 三、实验验证与结果分析

### 3.1 消融实验

![3](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201057.png)

1. Adaptive：保留 MegaCRN 的 GCRN 编码器 - 解码器，让编码器和解码器共享一个自适应图
2. Momentary：从 MegaCRN 中去掉 Meta - Node Bank，直接使用 Hyper - Network（即 FC 层）根据编码器的隐藏状态生成瞬时图供解码器使用。
3. Memory：从 MegaCRN 中去掉 Hyper - Network，仅使用 Memory Network（即 Meta - Node Bank）获取增强的隐藏状态，与编码器共享相同的自适应图

### 3.2 效率研究

![4](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201131.png)

### 3.3 定性研究

#### 3.3.1 时空解缠能力验证

![5](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201148.png)
通过 t-SNE 算法将节点嵌入可视化到低维空间，对比自适应图结构学习（adaptive GSL）和元图（meta-graph）的效果。结果显示，元图能够自动对节点（道路链接）进行聚类，并且随着时间从 t 到 t+1 演变，聚类效果持续存在但聚类形状发生变化。这一现象表明模型具备时空解缠能力，即可以区分不同时空模式下的道路链接。

同时也证明了模型具有时间适应性。通过映射不同聚类中道路链接的物理位置，并结合其日平均时间序列图，发现不同聚类在交通模式上存在显著差异，进一步验证了 Meta-Graph Learner 能够明确区分时空异质性。

#### 3.3.2 事件感知能力验证

![6](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201202.png)

选取一个发生在特定时间和道路链接上的交通事件案例，对比 MegaCRN 与 GW-Net、CCRNN 的预测结果。在 60 分钟的预测提前期内，MegaCRN 不仅能更好地捕捉正常的交通波动，还能适应包括高峰时段和交通事故等复杂情况，而其他模型在面对突发事件时会出现检测延迟或失败的问题。

通过可视化模型在不同情况下对 Meta-Node Bank 的模式查询权重，发现正常情况、高峰时段和事故情况下的查询权重存在差异，这表明模型能够区分不同的交通场景，具有较强的泛化能力。此外，可视化学习到的局部元图发现，元图会随着时间和交通事件的发生而变化，进一步验证了模型的优越适应性。
