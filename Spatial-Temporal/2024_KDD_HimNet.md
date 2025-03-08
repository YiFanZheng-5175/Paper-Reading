>领域：时空序列预测  
>发表在：KDD 2024  
>模型名字：***H***eterogeneity-***I***nformed ***M***eta-Parameter ***Net***work  
>文章链接：[Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting](https://arxiv.org/abs/2405.10800)  
>代码仓库：[https://github.com/XDZhelheim/HimNet](https://github.com/XDZhelheim/HimNet)  
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308154752.png)
# 一、研究背景与问题提出
## 1. 1 研究现状
为了考虑不同的情境，基于图的方法试图通过利用手工制作的特征（例如图拓扑和时间序列之间的相似性度量）来捕获异质性。然而，严重依赖预定义的表示限制了它们的适应性和通用性，因为这些特征不能涵盖不同情境的全部复杂性。为了提高适应性，其他工作引入了元学习技术，这些技术在不同的空间位置应用多个参数集。***然而，它们也依赖于辅助特征，包括兴趣点（POI）和传感器位置。*** 此外，高计算和内存成本限制了它们在大规模数据集上的适用性。最近的表示学习方法通过输入嵌入有效地识别异质性。然而，它们***过于简化的下游处理结构***未能充分利用表示能力。虽然自监督学习方法也通过设计额外的任务成功地捕获了时空异质性，但对于这些方法来说，***时空预测的完全端到端联合优化仍然存在困难***。  

简而言之，关键缺点是：  
1. 依赖辅助特征  
2. 高计算和内存成本；
3. 未能充分利用捕获的异质性；
4. 端到端优化困难。  

# 二、问题剖析与解决策略
## 2.1 解决方法
为了解决先前工作中的上述局限性，我们提出了一种新颖的异质性信息元参数学习方案和一个用于时空时间序列预测的异质性信息元网络（HimNet）。具体而言，我们***首先从聚类的角度通过学习空间和时间嵌入来隐含地表征异质性***。在训练过程中，这些表示逐渐区分并形成不同的聚类，捕捉潜在的时空上下文。接下来，***提出了一种新颖的时空元参数学习范式***，以提高模型的适应性和泛化能力，适用于各种领域。具体来说，我们通过查询一个小型元参数池为每个时空上下文学习一个独特的参数集。

基于这些，我们进一步提出了***异质性感知元参数学习***，它利用表征的异质性来指导元参数学习。因此，我们的方法不仅可以捕捉而且可以明确地利用时空异质性来改进预测。最后，我们设计了一个名为 HimNet 的端到端网络，实现了这些用于时空预测的技术。
### 2.1.1 时空异质性建模
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308162945.png)
### 2.1.2 时空元参数学习
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308163021.png)
### 2.1.3 异质性信息时空元网络
总而言之，把点子结合进了AGRU的backbone，做了一个Encoder,Decoder 

## 2.2 模型结构
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308154752.png)
# 三、实验验证与结果分析 
### 3.1 消融实验
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308164021.png)

1. 无 $E_{t}$：用全为 1 的静态矩阵替换时间嵌入，从而去除时间异质性建模。  
2. 无 $E_{s}$：用全为 1 的静态矩阵替换空间嵌入，从而去除空间异质性建模。  
3. 无 $E_{s t}$：用全为 1 的静态矩阵替换时空嵌入。
4. 无 $TMP$：通过降级为随机初始化的参数来去除时间元参数。
5. 无 $SMP$：以同样的方式去除空间元参数。
6. 无 $STMP$：以同样的方式去除时空混合元参数。

### 3.2 效率研究
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308164459.png)  

我们进一步分析了一个变体 HimNet- $\Theta'$，它去除了元参数池并直接优化了三个扩大的参数空间。HimNet- $\Theta'$ 包含超过 109 亿个参数，在我们能得到的任何 GPU 上都不可用。

### 3.3 案例研究

![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308164733.png)

![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308164836.png)

![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308164908.png)

