领域：时空序列预测  
发表在：ICDE 2023  
模型名字：***S***patio-***T***emporal ***Wave***lets  
文章链接：[When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks](https://ieeexplore.ieee.org/document/10184591)  
代码仓库：[https://github.com/LMissher/STWave](https://github.com/LMissher/STWave)  
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2023_ICDE_STWave-20250305214138.png]].png)
# 一、研究背景与问题提出
## 1. 1 研究现状
一系列交通预测方法单独为每个传感器的交通数据中的时间模式建模，如长短期记忆网络（LSTM）、时间卷积网络（TCN） 和 Transformer。然而，它们忽略了交通道路网络上传感器之间复杂的空间相关性，例如，一个传感器记录的交通量受到环境的影响。***（Comment：这也太古早了）***

另一系列方法进一步将图卷积网络（GCNs）与序列方法相结合，以同时捕获时空模式，如时空图卷积网络（STGCN）和扩散卷积循环神经网络（DCRNN）。
随后，为了消除根据交通道路网络预定义图的影响，图小波网络（GWN）和自适应图卷积循环网络（AGCRN）在图卷积网络中用自适应图替代预定义图，通过反向传播捕获交通数据中的全局和准确空间依赖关系。同时，它们失去了先验知识的指导，容易出现欠拟合或过拟合 。

与它们相比，时空融合图神经网络（STFGNN）可以通过时空融合图有效地利用交通道路网络中的结构和语义先验知识以及历史交通值。因此，基于 STFGNN，时空图神经 ODE（STGODE）利用基于张量的神经 ODE 来缓解深度图卷积中的过平滑问题 。

尽管一些方法提出了通用方法，如插件网络和协方差损失来提高基于图卷积网络模型的性能。然而，大多数基于图卷积网络的方法忽略了道路网络上传感器之间的相关性随时间不断变化。***（Comment：同年的AAAI就有这方面的研究）***
## 1.2 现存问题
1. 时间序列方面：
	交通时间序列复杂，包含多个局部独立模块，容易导致端到端的时空网络（STNet）过拟合。而且当其中某个模块出现***分布偏移***时，***端到端的 STNets 难以处理***，使得学习到的预测关联在非平稳交通时间序列上泛化性差
2. 空间相关性方面
	***基于图的深度学习方法在捕捉交通数据空间相关性时存在缺陷***。例如，GCN 基模型无法捕捉时变空间相关性 ***（Coment: 可是不是有循环网络结合GCN的研究吗？）***；GAT 基模型只能动态捕捉邻居间的空间相关性；全 GAT 模型虽然能动态捕捉所有传感器间的空间相关性，但计算复杂度高达 $O(N^2)$ ，在大规模数据集上计算负担重，并且该方法仅考虑基于值的空间语义信息，缺乏图结构信息，容易过拟合。
## 1.3 引出思考
1. 能否解决分布偏移的问题？
2. 能否解决空间相关性是随时间变化的问题？
3. 能否进一步增强空间相关性的建模？
# 二、问题剖析与解决策略
## 2.1 解决方法
### 2.1.1 解耦融合框架
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306204408.png)

利用离散小波变换（DWT）将复杂的交通数据解耦为稳定趋势和波动事件，之后通过双通道时空网络分别对趋势和事件进行建模，最后融合两者信息进行交通预测，以此缓解分布偏移问题，提高模型在非平稳交通数据上的适应性。
### 2.1.2 设计合适的 STNet
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306204434.png)
***（Comment：这也太复杂了）***
为 STWave 设计专门的时空网络，在事件建模中采用TCN捕捉***波动***的时间变化；在趋势建模中运用具有全局时间感受野的Attention捕捉***稳定***的时间变化；同时使用全 GAT 捕捉基于不同时间变化的动态全局空间相关性。
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306204606.png)
此外，还创新性地采用了查询采样策略(一个降低Attention复杂度的策略)和图小波基图位置编码（一种的位置编码），降低全 GAT 的复杂度并提升其结构感知能力。
### 2.1.3 多监督与自适应融合
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306204450.png)
在多监督解码器中，不仅对交通流量或速度进行监督，还添加对稳定趋势的辅助损失，以更好地处理事件中的分布偏移问题。同时，设计自适应事件融合模块（Cross attention），通过注意力机制自适应地判断并融合事件中的有用信息，提高预测的准确性。
（Comment：corss attention融合，同时监督trend的生成）
## 2.3 模型结构
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2023_ICDE_STWave-20250305214138.png]].png)
# 三、实验验证与结果分析 
### 3.1 消融实验
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306204636.png)
- **w/o DF**：去掉解耦流层，直接将交通数据输入模型，遵循端到端范式。该变体模型性能不如 STWave，因为它忽略了对交通时间序列中独立组件的解耦，可能导致过拟合。
- **w/o AF**：用加法操作代替自适应融合，直接将事件和趋势相加，忽略了对事件的虚假预测。此变体模型性能低于 STWave，说明添加自适应融合模块可以去除不正确的事件信息，对模型性能提升有积极作用。
- **w/o MS**：去掉趋势监督，模型结果仅由交通数据监督，缺乏平稳监督信号，可能导致不合理的预测。实验显示该变体模型性能不如 STWave，表明趋势监督有助于模型更好地处理事件中的分布偏移问题，提高预测的合理性。
- **w/o Tem**：去掉时间神经网络，使模型无法捕捉时间变化。该变体模型在大多数任务上的表现比去掉空间组件的变体模型要好，说明在多变量交通预测任务中，空间维度比时间维度对模型性能的影响更大。
- **w/o Spa**：去掉 ESGAT（Efficient Spectral Graph Attention Network），模型无法捕捉空间相关性。该变体模型性能明显下降，进一步证明了 ESGAT 在捕捉空间相关性方面的重要性，以及空间维度在多变量交通预测中的关键作用。
### 3.2 ESGAT的有效性研究
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306204659.png)
为展示 ESGAT 的有效性和效率，将 STWave 与基于注意力的 LSGCN、最先进的基线模型 STGODE 和 STFGNN，以及一种 STWave 的变体 “Full”（即 STWave 去掉 ESGAT 中的查询采样策略，计算所有空间相关性）进行对比。
### 3.3 图小波位置编码的有效性研究
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2023_ICDE_STWave-20250305222259.png]].png)
为验证图小波位置编码的有用性，提出了三种 STWave 模型的变体：“w/o GPE”（不再使用图位置编码）、“EV”（利用图拉普拉斯特征向量作为图位置编码）、“N2V”（使用 Node2vec 学习局部感知图位置编码）。