# TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis

>领域：时间序列预测  
>发表在：ICLR 2023  
>模型名字：TimesNet  
>文章链接：[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://openreview.net/forum?id=ju_Uqw384Oq)  
>代码仓库：[Time-Series-Library](https://github.com/thuml/Time-Series-Library)  
![20250401091822](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250401091822.png)

## 一、研究背景与问题提出

### 1.1 研究现状

特别是在深度学习社区中，受益于深度模型强大的非线性建模能力，许多工作被提出以捕捉现实世界时间序列中的复杂时间变化。一类方法采用循环神经网络（RNN）基于马尔可夫假设对连续时间点进行建模（Hochreiter & Schmidhuber，1997；Lai 等人，2018；Shen 等人，2020）。然而，这些方法通常在捕获长期依赖关系方面表现不佳，并且其效率受到顺序计算范式的影响。另一类方法利用沿时间维度的卷积神经网络（TCN）提取变异信息（Franceschi 等人，2019 年；He 和 Zhao，2019 年）。此外，由于一维卷积核的局部性，它们只能对相邻时间点之间的变化进行建模，因此仍然无法处理长期依赖关系。最近，具有注意力机制的 Transformer 在序列建模中得到了广泛应用（Brown 等人，2020 年；Dosovitskiy 等人，2021 年；Liu 等人，2021b）。在时间序列分析中，许多基于 Transformer 的模型采用注意力机制或其变体来捕获时间点之间的成对时间依赖关系（Li 等人，2019 年；Kitaev 等人，2020 年；Zhou 等人，2021 年；2022 年）。**但是注意力机制很难直接从分散的时间点中找出可靠的依赖关系，因为时间依赖关系可能会在复杂的时间模式中被深深掩盖**（Wu 等人，2021 年）。

## 二、问题剖析与解决策略

在本文中，为了解决复杂的时间变化问题，我们从多周期性的新维度分析时间序列。首先，我们观察到现实世界中的时间序列通常呈现出多周期性，例如天气观测的日变化和年变化，电力消耗的周变化和季度变化。这些多个周期相互重叠和相互作用，使得变化建模变得棘手。其次，对于每个周期，我们发现每个时间点的变化不仅受到其邻近区域的时间模式的影响，而且与相邻周期的变化高度相关。为了清晰起见，我们将这两种类型的时间变化分别命名为周期内变化和周期间变化。前者表示一个周期内的短期时间模式。后者可以反映连续不同周期的长期趋势。请注意，对于没有明显周期性的时间序列，变化将由周期内变化主导，并且等同于具有无限周期长度的时间序列的变化。

![20250401085742](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250401085742.png)

由于不同的时期会导致不同的期内和期间变化，多周期性自然可以推导出一种用于时间变化建模的模块化架构，在其中我们可以在一个模块中捕获由特定时期产生的变化。此外，这种设计使复杂的时间模式得以解开，有利于时间变化建模。然而，值得注意的是，一维时间序列很难同时明确呈现两种不同类型的变化。为了克服这个障碍，我们将时间变化的分析扩展到二维空间。具体来说，如图 1 所示，我们可以将一维时间序列重塑为二维张量，其中每列包含一个周期内的时间点，每行涉及不同周期中相同阶段的时间点。因此，通过将一维时间序列转换为一组二维张量，我们可以打破原始一维空间中表示能力的瓶颈，并成功地在二维空间中统一期内和期间变化，获得时间二维变化。从技术上讲，基于上述动机，我们超越了以前的骨干网络，提出 TimesNet 作为一种用于时间序列分析的新的任务通用模型。在 TimesBlock 的支持下，TimesNet 可以发现时间序列的多周期性，并在模块化架构中捕获相应的时间变化。具体来说，TimesBlock 可以根据学习到的周期自适应地将一维时间序列转换为一组二维张量，并通过参数高效的初始块进一步捕获二维空间中的期内和期间变化。

### 2.1 解决方法

#### 2.1.1 将一维变化转换为二维变化

![20250401085742](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250401085742.png)

如图1所示，每个时间点同时涉及其相邻区域以及不同周期中相同相位的两种时间变化，即周期内变化和周期间变化。然而，时间序列的这种原始一维结构只能表示相邻时间点之间的变化。为了解决这个限制，我们探索了时间变化的二维结构，它可以明确表示周期内和周期间的变化，从而在表示能力上更具优势，并有利于后续的表示学习。

具体来说，对于记录了$C$个变量、长度为$T$的时间序列，其原始一维组织形式为$\mathbf{X}_{1\text{D}} \in \mathbb{R}^{T \times C}$ 。为了表示周期间变化，我们首先需要发现周期。从技术上讲，我们通过快速傅里叶变换（FFT）在频域中对时间序列进行分析，如下所示：
$$
\mathbf{A} = \text{Avg}\left( \text{Amp}\left( \text{FFT}(\mathbf{X}_{1\text{D}}) \right) \right), \; \{ f_1, \cdots, f_k \} = \underset{f_* \in \{ 1, \cdots, \lfloor \frac{T}{2} \rfloor \} }{\text{arg Topk}} \; (\mathbf{A}), \; p_i = \left\lceil \frac{T}{f_i} \right\rceil, \; i \in \{ 1, \cdots, k \}.
\tag{1}
$$

这里，$\text{FFT}(\cdot)$ 和 $\text{Amp}(\cdot)$ 分别表示快速傅里叶变换和幅值计算。$\mathbf{A} \in \mathbb{R}^{T}$ 表示每个频率计算出的幅值，它是通过 $\text{Avg}(\cdot)$ 对$C$个维度求平均得到的。注意，第$j$个值 $\mathbf{A}_j$ 表示频率为$j$的周期基函数的强度，对应于周期长度 $\left\lceil \frac{T}{j} \right\rceil$ 。考虑到频域的稀疏性，并且为了避免无意义的高频带来的噪声（Chatfield, 1981; Zhou等人, 2022），我们只选择幅值最大的前$k$个值，并得到最显著的频率 $\{ f_1, \cdots, f_k \}$ 及其未归一化的幅值 $\{ \mathbf{A}_{f_1}, \cdots, \mathbf{A}_{f_k} \}$ ，其中$k$是超参数。这些选定的频率也对应着$k$个周期长度 $\{ p_1, \cdots, p_k \}$ 。由于频域的共轭性，我们只考虑 $\{ 1, \cdots, \lfloor \frac{T}{2} \rfloor \}$ 范围内的频率。我们将公式(1)总结如下：
$$
\mathbf{A}, \{ f_1, \cdots, f_k \}, \{ p_1, \cdots, p_k \} = \text{Period}(\mathbf{X}_{1\text{D}}).
\tag{2}
$$

基于选定的频率 $\{ f_1, \cdots, f_k \}$ 和相应的周期长度 $\{ p_1, \cdots, p_k \}$ ，我们可以通过以下公式将一维时间序列 $\mathbf{X}_{1\text{D}} \in \mathbb{R}^{T \times C}$ 重塑为多个二维张量：
$$
\mathbf{X}_{2\text{D}}^i = \text{Reshape}_{p_i, f_i}(\text{Padding}(\mathbf{X}_{1\text{D}})), \; i \in \{ 1, \cdots, k \},
\tag{3}
$$
其中 $\text{Padding}(\cdot)$ 是通过在时间维度上用零扩展时间序列，使其与 $\text{Reshape}_{p_i, f_i}(\cdot)$ 兼容，其中$p_i$ 和$f_i$ 分别表示变换后的二维张量的行数和列数。注意，$\mathbf{X}_{2\text{D}}^i \in \mathbb{R}^{p_i \times f_i \times C}$ 表示基于频率$f_i$ 重塑后的第$i$个时间序列，其列和行分别表示在相应周期长度$p_i$ 下的周期内变化和周期间变化。最终，如图2所示，基于选定的频率和估计的周期，我们得到一组二维张量 $\{ \mathbf{X}_{2\text{D}}^1, \cdots, \mathbf{X}_{2\text{D}}^k \}$ ，这表示由不同周期导出的$k$种不同的时间二维变化。

同样值得注意的是，这种变换为变换后的二维张量带来了两种局部性，即相邻时间点之间的局部性（列，周期内变化）和相邻周期之间的局部性（行，周期间变化）。因此，时间二维变化可以很容易地由二维卷积核处理。

#### 2.1.2 TimesBlock

![20250401091822](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250401091822.png)

如图3所示，我们以残差方式构建TimesBlock（He等人，2016）。具体来说，对于长度为$T$的一维输入时间序列$\mathbf{X}_{1\text{D}} \in \mathbb{R}^{T \times C}$ ，我们在一开始通过嵌入层$\mathbf{X}_{1\text{D}}^0 = \text{Embed}(\mathbf{X}_{1\text{D}})$ 将原始输入投影到深度特征$\mathbf{X}_{1\text{D}}^0 \in \mathbb{R}^{T \times d_{\text{model}}}$ 。对于TimesNet的第$l$层，输入是$\mathbf{X}_{1\text{D}}^{l - 1} \in \mathbb{R}^{T \times d_{\text{model}}}$ ，该过程可以形式化为：
$$
\mathbf{X}_{1\text{D}}^l = \text{TimesBlock}(\mathbf{X}_{1\text{D}}^{l - 1}) + \mathbf{X}_{1\text{D}}^{l - 1}.
\tag{4}
$$

如图3所示，对于第$l$个TimesBlock，整个过程涉及两个连续部分：捕捉时间二维变化，以及自适应聚合来自不同周期的表示。

**捕捉时间二维变化** 与公式(1)类似，我们可以通过$\text{Period}(\cdot)$ 估计深度特征$\mathbf{X}_{1\text{D}}^{l - 1}$ 的周期长度。基于估计的周期长度，我们可以将一维时间序列转换到二维空间并得到一组二维张量，从中我们可以通过参数高效的Inception模块方便地获得信息表示。该过程形式化如下：
$$
\begin{align*}
\mathbf{A}^{l - 1}, \{ f_1, \cdots, f_k \}, \{ p_1, \cdots, p_k \} &= \text{Period}(\mathbf{X}_{1\text{D}}^{l - 1}), \\
\mathbf{X}_{2\text{D}}^{l, i} &= \text{Reshape}_{p_i, f_i}(\text{Padding}(\mathbf{X}_{1\text{D}}^{l - 1})), \; i \in \{ 1, \cdots, k \} \\
\tilde{\mathbf{X}}_{2\text{D}}^{l, i} &= \text{Inception}(\mathbf{X}_{2\text{D}}^{l, i}), \; i \in \{ 1, \cdots, k \} \\
\tilde{\mathbf{X}}_{1\text{D}}^{l, i} &= \text{Trunc}(\text{Reshape}_{1, (p_i \times f_i)}(\tilde{\mathbf{X}}_{2\text{D}}^{l, i})), \; i \in \{ 1, \cdots, k \},
\end{align*}
\tag{5}
$$
其中$\mathbf{X}_{2\text{D}}^{l, i} \in \mathbb{R}^{p_i \times f_i \times d_{\text{model}}}$ 是第$i$个变换后的二维张量。变换后，我们通过参数高效的Inception模块（Szegedy等人，2015）将其作为$\text{Inception}(\cdot)$ 处理二维张量，该模块涉及多尺度二维卷积核，并且是最著名的视觉主干之一。然后我们将学习到的二维表示$\tilde{\mathbf{X}}_{2\text{D}}^{l, i}$ 转换回一维$\tilde{\mathbf{X}}_{1\text{D}}^{l, i} \in \mathbb{R}^{T \times d_{\text{model}}}$ 以进行聚合，其中我们采用$\text{Trunc}(\cdot)$ 来截断长度为$p_i \times f_i$ 的序列到原始长度$T$ 。

注意，受益于一维时间序列的变换，Inception模块中的二维卷积核可以同时聚合多尺度的周期内变化（列）和周期间变化（行），涵盖不同的相邻时间点和相邻周期。此外，我们采用一个共享的Inception模块来处理不同的重塑二维张量$\{ \mathbf{X}_{2\text{D}}^{l, 1}, \cdots, \mathbf{X}_{2\text{D}}^{l, k} \}$ ，以提高参数效率，这可以使模型大小与超参数$k$的选择无关。

**自适应聚合** 最后，我们需要为下一层融合$k$个不同的一维表示$\{ \tilde{\mathbf{X}}_{1\text{D}}^{l, 1}, \cdots, \tilde{\mathbf{X}}_{1\text{D}}^{l, k} \}$ 。受自相关（Wu等人，2021）的启发，幅值$\mathbf{A}$ 可以反映所选频率和周期的相对重要性，从而对应于每个变换后的二维张量的重要性。因此，我们基于幅值聚合一维表示：
$$
\begin{align*}
\tilde{\mathbf{A}}_{f_1}^{l - 1}, \cdots, \tilde{\mathbf{A}}_{f_k}^{l - 1} &= \text{Softmax}(\mathbf{A}_{f_1}^{l - 1}, \cdots, \mathbf{A}_{f_k}^{l - 1}) \\
\mathbf{X}_{1\text{D}}^l &= \sum_{i = 1}^{k} \tilde{\mathbf{A}}_{f_i}^{l - 1} \times \tilde{\mathbf{X}}_{1\text{D}}^{l, i}.
\end{align*}
\tag{6}
$$

由于周期内和周期间的变化已经包含在多个高度结构化的二维张量中，TimesBlock可以同时充分捕捉多尺度时间二维变化。因此，与直接从一维时间序列学习相比，TimesNet可以实现更有效的表示学习。

**二维视觉主干的通用性** 受益于将一维时间序列转换为时间二维变化，我们可以选择各种广泛使用的视觉主干来替代Inception模块进行表示学习，如流行的ViT（He等人，2016）和ResNeXt（Xie等人，2017）、先进的ConvNeXt（Liu等人，2022b）以及基于注意力的模型（Liu等人，2021b）。因此，我们的二维时间设计也将一维时间序列与蓬勃发展的二维视觉主干联系起来，利用了计算机视觉社区发展带来的优势。一般来说，更强大的用于表示学习的二维主干将带来更好的性能。考虑到性能和效率（图4右），我们基于参数高效的Inception模块进行主要实验，如公式(5)所示。

### 2.2 模型结构

![20250401091822](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250401091822.png)

## 三、实验验证与结果分析

### 3.1 消融实验

**表示分析** 我们尝试从表示学习的角度来解释模型性能。从图6中可以发现，预测和异常检测任务中更好的性能对应着更高的CKA相似性（2019），这与插补和分类任务的情况相反。需要注意的是，较低的CKA相似性意味着不同层之间的表示具有区分性，即分层表示。因此，这些结果也表明了每个任务所需的表示属性。如图6所示，TimesNet能够为不同任务学习合适的表示，例如在预测和异常检测中的重建任务中学习低级表示，在插补和分类任务中学习分层表示。相比之下，FEDformer（2022）在预测和异常检测任务中表现良好，但在学习分层表示方面表现不佳，导致在插补和分类任务中性能较差。这些结果也验证了我们提出的TimesNet作为基础模型的任务通用性。

![20250403132440](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250403132440.png)

**时间二维变化** 我们在图7中提供了一个关于时间二维变化的案例研究。我们可以发现，TimesNet能够精确捕捉多周期性。此外，变换后的二维张量具有高度结构化且信息丰富，其列和行分别可以反映时间点之间和周期之间的局部性，这支持了我们采用二维卷积核进行表示学习的动机。更多可视化内容见附录D。

![20250403132500](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250403132500.png)
