>领域：时空序列预测  
>发表在：CIKM 2022  
>模型名字：***S***patio-***T***emporal ***A***daptive ***E***mbedding Vanilla Trans***former***  
>文章链接：[STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting](https://arxiv.org/abs/2308.10425)  
>代码仓库：[https://github.com/XDZhelheim/STAEformer](https://github.com/XDZhelheim/STAEformer)  
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201301.png)
# 一、研究背景与问题提出
## 1. 1 研究现状
STGNNs和基于 Transformer 的模型因其出色的性能而非常受欢迎。研究人员投入了大量精力来开发用于交通预测的奇特而复杂的模型，例如新颖的图卷积、学习图结构 、高效的注意力机制 以及其他方法。
## 1.2 现存问题
尽管如此，网络架构的进步遇到了***性能提升递减***的情况，促使人们将***重点***从复杂的模型设计转向***数据本身的有效表示技术***。
## 1.3 引出思考
能否在***模型结构***之外，研究***数据本身的有效表示技术***，以进一步增强模型性能？
# 二、问题剖析与解决策略
## 2.1 解决方法
### 2.1.1 时空自适应嵌入矩阵
 $E_{a} \in \mathbb{R}^{T ×N ×d_{a}}$
## 2.2 模型结构
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201301.png)
# 三、实验验证与结果分析 
### 3.1 消融实验
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201349.png)  
$w/o$ $E_{a}$ 移除时空嵌入（实际上就是node emb）  $E_{a}$ .  
$w/o$ $E_{p}$  移除周期性嵌入 $E_{p}$  
$w/o$ 𝑇-𝑇𝑟𝑎𝑛𝑠. 移除T transformer层  
$w/o$ 𝑆𝑇-𝑇𝑟𝑎𝑛𝑠. 移除temporal transformer 层   和 spatial transformer 层

### 3.2 案例学习
#### 3.2.1 与Node Emb比较
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201419.png)

#### 3.2.2 时空嵌入的可视化
![](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250306201429.png)