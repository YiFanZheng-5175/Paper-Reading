# Get Rid of Isolation: A Continuous Multi-task Spatio-Temporal Learning Framework

>领域：时空序列预测  
>发表在：NeurIPS 2024  
>模型名字：***C***ontinuous ***Mu***lti-task ***S***patio-***T***emporal  
>文章链接：[Get Rid of Isolation: A Continuous Multi-task Spatio-Temporal Learning Framework](https://arxiv.org/abs/2410.10524)  
>代码仓库：[https://github.com/DILab-USTCSZ/CMuST](https://github.com/DILab-USTCSZ/CMuST)  
![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308165602.png)  

## 一、研究背景与问题提出

### 1. 1 研究现状

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308170109.png)

图 1 中，交通流量模式会随着城市扩张和新兴趣点的建立而演变。同时，随着对道路安全的日益关注，交通事故预测已成为智能交通中的一项新任务，但不可避免地会遇到冷启动问题。

不幸的是，***传统的特定任务模型通常假设单个任务的数据遵循独立同分布，并且在数据密集可用的情况下***，这种假设直接导致在数据稀疏场景下的失败以及对新任务的泛化能力不足。实际上，对于不同的数据集，分别训练单个特定任务的时空模型成本较高，并且会使模型陷入孤立。

为此，我们认为一个连续的多任务时空学习框架对于促进任务级合作是非常必要的。***通过多任务学习共同对多领域数据集进行建模更加有趣和令人兴奋***，这使得我们能够从整体角度理解时空系统，并通过利用来自不同数据领域的集体智慧来强化每个单独的任务。

利用任务间相关性实现相互提升的关键在于捕捉数据维度和领域之间的共同相互依赖关系。***当前的多任务学习方案要么研究辅助任务和主要任务之间的正则化效应，要么设计损失目标以约束每个任务之间的一致性***。实际上，给定一个时空域，不同数据类型和领域之间必然存在共同的相互依赖关系，这对于协同学习是有价值的。

即使多任务学习和时空预测蓬勃发展，也从未有一个系统的解决方案来解决不同任务的各种来源数据如何通过多任务学习增强特定任务。更具体地说，系统中观测值之间的相互关系可以分解为多维相互作用，即从上下文环境到各自的空间关系和时间演化、时空层面的相互作用以及不同数据领域之间的关系。

考虑到多层次的语义相关性，需要从整体角度学习系统，这进一步给连续的多任务时空学习框架带来了两个挑战，即：1）***如何解开数据维度和领域之间的复杂关联，并以自适应方式捕捉这种依赖关系以改进时空表示***，从而促进共同模式的提取以实现相互增强。2）***如何利用任务级别的共性和个性来联合建模多任务数据集，并利用这种提取的任务级别的共性和多样性来强化各个任务以摆脱任务孤立***

## 二、问题剖析与解决策略

在我们的工作中，提出了一种连续多任务时空学习框架 CMuST，以联合对集成城市系统中的多个数据集进行建模，从而增强各自的学习任务。

具体来说，首先设计了一个多维时空交互网络（MSTI），以剖析数据维度之间的交互，包括上下文 - 空间、上下文 - 时间以及空间和时间维度内的自交互。MSTI 能够通过交互改进时空表示，并提供***解缠模式***以支持***共性提取***。

之后，提出了一种滚动适应训练方案 RoAda，它迭代地捕获任务方面的一致性和任务特定的多样性。在 RoAda 中，为了***保持任务特征***，构建了一个特定于任务的提示，通过自动编码器压缩数据模式来***保留与其他任务区分开的独特模式***。为了***捕获跨任务的共性***，我们提出了一种权重行为建模策略，以迭代地突出可学习权重的最小变化，即在连续训练期间的稳定交互，它封装了关键的任务级共性。这种方法不仅通过连续的任务滚动稳定学习，而且通过共享模式减轻新任务的冷启动问题。***最后，设计了一种特定于任务的细化方法，以利用共性并对特定任务进行细粒度适应***。

### 2.1 解决方法

#### 2.1.1 多维时空交互网络

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308171328.png)

- **空间-上下文交叉交互**：采用多头交叉注意力（MHCA）架构，让空间和观测分量交替作为查询（Q）和键值（KV）对。通过特定的变换生成查询、键和值，计算注意力分数后，将嵌入送入前馈网络（FFN）增强学习能力，最后经层归一化（LN）处理，将结果与原特征在相应维度拼接，以此丰富原始表示，使其包含更精细的跨维度关系。
- **时间-上下文交叉交互**：先对表示进行转置，并引入步长位置编码，使注意力机制能够感知时间演化。后续步骤与空间-上下文交叉注意力机制类似，最终得到空间和时间维度交叉交互的结果。
- **空间和时间维度内的自交互**：先在时间维度上进行自注意力计算，计算过程涉及从整体表示中获取查询、键和值，输出经FFN和非线性变换、LN处理，得到时间自交互结果。将该结果张量转置后，以类似方式进行空间自交互计算，对空间交互进行细化并聚合空间节点特征。
- **融合策略与损失函数**：MSTI通过融合策略自适应地整合上述各种交互，采用Huber损失函数确保对时空样本中的异常值具有鲁棒性。这种设计使得MSTI能够有效提取多样的交互信息，包括时空域交互和各维度内的自交互，增强数据关系学习，为跨任务域的共性提取提供有力支持。

#### 2.1.2 滚动适应训练方案 RoAda

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308171338.png)

- **任务总结作为提示**：针对每个任务，利用采样-自动编码方案进行任务总结。先对任务的主要观测数据按每天相同时间进行平均采样，得到周期性样本表示，再将其输入自动编码器。通过自动编码器的编码和解码过程，提取压缩且有区分度的数据模式，这些模式封装了任务的核心特征。将编码后的特征进一步变换为任务特定提示，用于在学习过程中区分不同任务的个性化特征。
- **权重行为建模**：这是RoAda的关键阶段，用于捕捉任务间的共性。首先独立训练模型学习任务 $T_1$，直至性能稳定，记录此时的模型权重 $W_{c}^{(T_{1})}$ 。然后模型切换到任务 $T_2$，加载任务 $T_2$的提示和数据集进行训练，在训练过程中，仔细存储模型权重的演化行为。通过计算权重在不同训练迭代中的方差，引入集体方差 $\circ$和阈值 $\delta$来区分稳定权重和动态权重。稳定权重 $W_{stable }^{(T_{2})}$被冻结，模型以稳定后的权重为初始化，继续学习后续任务。重复此过程，直至完成所有任务的学习，最终得到包含任务间共性模式的稳定权重，增强模型的泛化能力。
- **任务特定细化阶段**：在完成权重行为建模后，进入任务特定细化阶段。在此阶段，模型利用前一阶段得到的稳定权重 $W^{*}$和任务特定提示  $P^{(T_{i})}$，对每个任务进行微调。通过持续训练，使模型在保持整体稳定性的同时，能够更好地适应每个任务的独特模式，将个体智能融入集成模型，提升模型在每个具体任务上的性能。

### 2.2 模型结构

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308170109.png)

## 三、实验验证与结果分析

### 3.1 消融研究

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308173555.png)

- “w/o context-data interaction”，去除 MSTI 模块中的空间 - 上下文和时间 - 上下文交叉交互，以此探究从上下文环境到空间关系和时间演化的多维交互对预测的作用
- 二是 “w/o consistency maintainer”，在 RoAda 阶段省略稳定权重的分离和冻结操作，改为使用所有权重进行滚动训练，从而验证捕捉任务一致性和共性对促进任务学习的有效性
- 三是 “w/o task-specific preserver”，去除任务特定提示，消除任务特定多样性的保留，观察其对模型性能的影响。

### 3.2 案例分析

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308173617.png)

1. **可视化注意力变化**：在RoAda阶段，对CMuST的注意力权重变化进行可视化。结果显示，随着任务的持续学习，不同维度之间的关系和交互变得更加清晰且稳定。这表明通过对权重行为进行建模，CMuST能够有效巩固维度级别的关系，进而实现跨任务的时空交互一致性提取。这种可视化分析直观地展示了模型在学习过程中如何逐渐捕捉和稳定不同维度之间的联系，为理解模型的学习机制提供了重要依据。
2. **任务增加时的性能变化**：以NYC数据集为例，研究随着任务数量增加，单个任务的性能变化情况。实验结果表明，随着更多任务的加入，每个任务的性能都有所提升。这一现象说明在CMuST框架下，任务不再相互孤立，而是能够通过吸收共同的表示和交互信息，获取集体智能，实现相互促进和提升。这充分体现了CMuST框架在多任务学习中的优势，证明了其能够有效打破任务之间的隔离，提高模型在各个任务上的性能表现。
