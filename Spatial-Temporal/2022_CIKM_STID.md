>领域：时空序列预测  
>发表在：CIKM 2022  
>模型名字：***S***patial-***T***emporal ***Id***entity  
>文章链接：[Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting](https://arxiv.org/abs/2208.05233)  
>代码仓库：[BasicTS/baselines/STID](https://github.com/GestaltCogTeam/BasicTS/tree/master/baselines/STID)  
![20250306185041](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306185041.png)

# 一、研究背景与问题提出
## 1. 1 研究现状
STGNNs使用GCN来处理非欧几里得依赖关系，使用序列模型来捕获时间模式。目前的研究一直致力于设计强大的图卷积，或者减少对预定义图结构的依赖（意思是减少对输入数据里邻接矩阵的依赖）。近期的STGNNs变得愈发***复杂***，但***性能提升有限***
## 1.2 现存问题
1. 空间和时间维度上样本的不可区分性
	简单的MLPs无法根据相似的历史数据预测它们不同的未来数据，也就是说，它们无法区别这些样本。STGNNs成功的关键因素之一是图卷积网络缓解了空间上的不可区分性。
	![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306195835.png)
## 1.3 引出思考
能否提炼出多元时间序列预测的关键因素，并设计一个与STGNNs一样强大，但更***简洁高效***的模型？
# 二、问题剖析与解决策略
## 2.1 解决方法
### 2.1.1 时空标识嵌入矩阵
STID使用一个空间嵌入矩阵 $\mathbf{E} \in \mathbb{R}^{N \times D}$，以及两个时间嵌入矩阵 $\mathbf{T}^{\mathrm{iD}} \in \mathbb{R}^{N_d \times D}$ 和 $\mathbf{T}^{\mathrm{iW}} \in \mathbb{R}^{N_w \times D}$ 来表示空间和时间标识。 $N$是变量的数量（即时间序列的数量）， $N_d$ 是一天中的时间片数量（由采样频率决定）， $N_w = 7$ 是一周中的天数， $D$ 是隐藏层维度。结果表明，***通过解决样本的不可区分性问题***，我们可以设计出更高效且有效的模型，***而不受时空图神经网络的限制***。
## 2.2 模型结构
![20250306185041](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306185041.png)
# 三、实验验证与结果分析 
### 3.1 消融实验
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306195926.png)所有这些标识矩阵都是有益的。其中***最重要的是空间标识***$\mathbf{E} \in \mathbb{R}^{N \times D}$，这意味着空间上的不可区分性是MTS预测的一个主要瓶颈。此外，时间标识 $T^{TID}$ 和 $T^{DiW}$ 也很重要，因为现实世界中的数据通常包含每日和每周的周期性。

### 3.2 可视化实验
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306195953.png)

1. ***首先***，(a)表明不同变量（即时间序列）的标识往往会聚类。这与交通系统的特征相符。例如，道路网络中相邻的交通传感器往往具有相似的模式。
2. ***其次***，(b)可视化了每天288个时间片的嵌入情况。很明显，PEMS08数据集中存在每日周期性。此外，相邻的时间片往往具有相似的标识。
3. ***最后***，(c)显示工作日的标识相似，而周末的标识则大不相同。