>领域：时空序列预测  
>发表在：VLDB 2024  
>模型名字：***Big*** ***S***patial***T***emporal  
>文章链接：[BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks](https://dl.acm.org/doi/abs/10.14778/3641204.3641217)  
>代码仓库：[https://github.com/usail-hkust/BigST.](https://github.com/usail-hkust/BigST)  
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2024_VLDB_BigST-20250302175945.png]].png)
# 一、研究背景与问题提出
## 1. 1 研究现状
近年来，时空图神经网络（STGNNs）通过将路网的潜在图结构作为归纳偏差纳入其中，在学习时空依赖关系方面显示出巨大的希望。
## 1.2 现存问题
1. STGNN 的计算和内存成本
	首先，未来的交通状态对长期历史观测产生复杂的时间依赖性，例如，重复出现的模式，将它们考虑在内有利于准确预测。
	尽管如此，STGNN 的计算和内存成本***随着输入序列长度的增加而爆炸式增长***，特别是在处理大规模道路网络时。
	为了减少计算开销，大多数现有模型***仅依赖短时间窗口***内（例如，过去一小时）的历史信息进行预测，这极大地限制了它们的性能。
	一些研究试图使用手工制作的特征对长期相关性进行建模，但它们***严重依赖于经验假设***，例如每日或每周周期性，以捕捉简单的周期性效应。
	这些方法无法对简化假设之外的复杂时间依赖性进行建模。
2. 学习潜在图结构所需的 $O(N^{2})$ 复杂度
	STGNN中固有的另一个本质问题是目标系统的拓扑结构，它直接决定了空间依赖建模的有效性和效率。  
	为了定义图拓扑，早期的研究简单地使用一些***预定义***的度量来计算节点之间的成对相似度，例如路网距离。  
	然而，在现实世界的场景中，目标交通系统的先验拓扑信息通常是不完整的、嘈杂的，并且可能偏向于特定的任务。  
	一个更有效的解决方案是在下游预测任务的监督下端到端学习一个潜在的图结构。例如，GWNET 学习一个密集的相邻矩阵 $A=\sigma(E_{1} E_{2}^{\top})$，通过可训练的节点嵌入矩阵 $E_{1}$和 $E_{2}$来捕获隐式节点交互。这样的方法减少了对先验知识的依赖，并且能够识别最优拓扑以便于学习更好的节点表示，从而获得最先进的性能。  
	然而，学习图结构需要 $O(N^{2})$计算复杂度，其中 $N$ 表示节点数，这阻碍了它们在大规模道路网络中的应用。  
## 1.3 引出思考
1. 如何高效和有效地利用***长期***历史***时间***序列上的知识？
	随着输入序列长度的增加，时间和空间复杂度急剧增长，模型很容易被噪声淹没，这对优化模型的效率和有效性提出了挑战
2. 如何降低学习***潜在图结构***中昂贵的***二次复杂度***？
	如前所述，现有的 STGNN 依赖于自适应图来捕获道路网络中任意节点之间的复杂空间依赖关系。然而，学习潜在图结构需要令人望而却步的二次计算复杂性，这对于扩展到巨大的道路网络具有挑战。
# 二、问题剖析与解决策略
## 2.1 解决方法
### 2.1.1 长序列特征提取器
一个在长时间步下预训练的线性Transformer Block 来高效和有效地利用***长期***历史***时间***序列上的知识，注意，此阶段不包含空间上的建模  
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308002229.png)  
#### 线性自注意力
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2024_VLDB_BigST-20250302181941.png]].png)  
#### 周期性特征采样
就是按照周期把x间隔采样出来

### 2.1.2 线性化全局空间卷积网络
将***潜在图结构***学习和大规模道路网络上的空间消息传递的***计算复杂度降低到线性***。
![[2024_VLDB_BigST-20250302182434.png]]
#### Patch-level 动态图学习
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2024_VLDB_BigST-20250302182520.png]].png)  
Patch-level 就是把特征D乘在T上面
#### 线性空间卷积
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2024_VLDB_BigST-20250302182538.png]].png)

## 2.2 模型结构
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2024_VLDB_BigST-20250302175945.png]].png)
# 三、实验验证与结果分析 
## 3.1 效率分析
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308003020.png)
## 3.2 消融实验
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2024_VLDB_BigST-20250302183202.png]].png)  
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2024_VLDB_BigST-20250302183218.png]].png)  

1. woCLT模型去除了由上下文感知线性化Transformer生成的预先计算的长期表示
2. woPFS模型排除了周期性特征
3. woSAC模型去除了空间卷积近似
4. woSE模型在图构建中仅使用动态嵌入
5. woDE模型在图构建中仅使用静态嵌入。结果如图4所示。
