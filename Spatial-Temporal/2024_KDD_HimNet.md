>领域：时空序列预测  
>发表在：KDD 2024  
>模型名字：***H***eterogeneity-***I***nformed ***M***eta-Parameter ***Net***work  
>文章链接：[Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting](https://arxiv.org/abs/2405.10800)  
>代码仓库：[https://github.com/XDZhelheim/HimNet](https://github.com/XDZhelheim/HimNet)  
![[2024_IJCAI_STDMAE-20250302170444.png]]
# 一、研究背景与问题提出
## 1. 1 研究现状
1. 时空预测
	早期主要依赖传统时间序列模型，后来 RNNs 和 CNNs 因能更好地捕捉复杂时间依赖关系而受到关注，但它们忽略了关键的空间相关性。为了联合捕捉时空特征，一些研究将图卷积网络（GCNs）与时间模型相结合，近年来还提出了许多新颖的时空模型，注意力机制也对时空预测产生了深远影响，一系列基于 Transformer 的模型展现出优越性能。
2. Mask Pre-train
	最近，许多研究人员尝试将预训练技术应用于时间序列数据。
## 1.2 现存问题
1. 时空预测
	然而，这些端到端模型***通常只关注短期输入***，难以捕捉完整的时空依赖关系，在区分时空异质性方面存在困难。
2. Mask Pre-train
	在时序数据上预训练的方法要么与***通道无关***，要么***忽略了空间维度的预训练***。
## 1.3 引出思考
能否学习到***清晰***的时空异质性和***完整***的时空异质性？
# 二、问题剖析与解决策略
## 2.1 问题瓶颈
### 2.1.1 时空异质性
![[2024_IJCAI_STDMAE-20250302172630.png]]
## 2.2 解决方法
### 2.2.1 Mask AutoEncoder Pre-training
![[2024_IJCAI_STDMAE-20250302173107.png]]

通过在Spatial和Temporal上进行Mask AutoEncoder 的 掩码重建训练，来学习***清晰***的时空异质性

### 2.2.2 Long
![[2024_IJCAI_STDMAE-20250302173300.png]]
通过把输入从短时间步改成长时间步来学习***完整***的时空异质性
## 2.3 模型结构
![[2024_IJCAI_STDMAE-20250302170444.png]]
# 三、实验验证与结果分析 
### 3.1 消融实验
![[2024_IJCAI_STDMAE-20250302173607.png]]
1. STD-MAE：分别在空间维度和时间维度上Mask
2. T-MAE：仅在时间维度上Mask
3. S-MAE：仅在空间维度上Mask
4. STM-MAE：使用 spatial-temporal-mixed Mask
5. w/o Mask：不使用Mask Pre-tarin
### 3.2 案例研究
#### 3.2.1 预训练中的重建精度
![[2024_IJCAI_STDMAE-20250302174313.png]]
#### 3.2.2 对时空Mirage的鲁棒性
![[2024_IJCAI_STDMAE-20250302174326.png]]