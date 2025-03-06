>领域：时空序列预测  
>发表在：ACM TKDD 2023  
>模型名字：***D***ynamic ***G***raph ***C***onvolutional ***R***ecurrent ***N***etwork  
>文章链接：[Dynamic Graph Convolutional Recurrent Network for Traffic Prediction: Benchmark and Solution](https://dl.acm.org/doi/10.1145/3532611)  
>代码仓库：[https://github.com/tsinghua-fib-lab/Traffic-Benchmark](https://github.com/tsinghua-fib-lab/Traffic-Benchmark)  
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306205343.png)
# 一、研究背景与问题提出
## 1. 1 研究现状
图神经网络的出现使深度学习模型能够处理非欧几里得数据，并且节点（传感器或路段）之间的道路距离通常用于计算边的权重 。此外，还提出了自适应邻接矩阵来保留隐藏的空间依赖性、。然而，***静态图很难动态地自我调整并捕获复杂的动态空间依赖性***。
CNN 也可以有效地捕获时间相关性 ，但由于其隐式时间建模而缺乏灵活性，这使得时间步长不可见。
自注意力机制可以有效地捕获全局依赖性，但不擅长短期预测。RNN 能够有力地捕捉序列相关性，但***相对耗时较多***。
## 1.2 现存问题
1. 动态特性建模不足
	预定义和自适应邻接矩阵都是静态的，难以反映道路网络拓扑的动态特性，且多数工作未能有效融合静态和动态图，同时避免过平滑问题。
2. RNN 效率受限
	RNN 及其变体在时间序列预测中应用广泛，但内部递归操作限制了训练速度，阻碍了如序列到序列架构在交通预测任务中的应用。
3. 缺乏公平比较
	随着交通预测领域的发展，模型和数据集不断增加，但模型在不同数据集和实验设置下评估，难以进行公平比较，阻碍了领域发展。
## 1.3 引出思考
解决上述问题的方法是什么？

# 二、问题剖析与解决策略
## 2.1 解决方法
### 2.1.1 动态图+静态图
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2023_TKDD_DGCRN-20250303222337.png]].png)
### 2.1.2 课程学习
RNN Decoder先在短的循环长度上进行训练
![[2023_TKDD_DGCRN-20250303222610.png]]
### 2.1.3 发布了新的数据集
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2023_TKDD_DGCRN-20250303222914.png]].png)
## 2.2 模型结构
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306205343.png)
# 三、实验验证与结果分析 
### 3.1 消融实验
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/![[2023_TKDD_DGCRN-20250303223025.png]].png)
1. w/o dg：去除动态邻接矩阵和动态图卷积。
2. w/o preA：去除预定义图卷积。
3. w/o hypernet：用简单全连接层替代动态图生成器中的超网络。
4. dg w/o time、dg w/o speed、dg w/o h：分别在生成动态图时不输入时间、速度、隐藏状态。
5. dg2sg：图不逐步更新，采用类似MTGNN的静态图生成方式
6. w/o cl：不使用课程学习策略，仅用预定采样训练。结果显示课程学习策略有效，能提升模型性能。
7. hypernet mul2matmul：将超网络中相关公式的哈达玛积替换为矩阵乘法。