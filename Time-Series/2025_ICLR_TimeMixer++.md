# TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis

>领域：时间序列预测  
>发表在：ICLR 2025  
>模型名字：TimeMixer++  
>文章链接：[TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis](https://arxiv.org/abs/2410.16032)  
>代码仓库：Coming Soon  
![20250326144426](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250326144426.png)

## 一、研究背景与问题提出

### 1.1 研究现状

一系列研究（Lai 等人，2018b；Zhao 等人，2017）利用循环神经网络（RNNs）来捕获序列模式。然而，由于马尔可夫假设和低效率等限制，这些方法通常难以捕获长期依赖关系。时间卷积网络（TCNs）（Wu 等人，2023；Wang 等人，2023a；Liu 等人，2022a；Wang 等人，2024a）有效地捕获局部模式，但在处理长期依赖关系时面临挑战（季节性和趋势性）。虽然一些方法基于频域信息将时间序列重塑为二维张量（Wu 等人，2023）或对时域进行下采样（Liu 等人，2022a），但它们在全面捕捉远程模式方面存在不足。相比之下，基于变压器的架构（Nie 等人，2023；Liu 等人，2024；Zhou 等人，2022b；Wang 等人，2022；Shi 等人，2024）利用标记级自注意力通过允许每个标记关注所有其他标记来对远程依赖关系进行建模，克服了固定感受野的局限性。然而，与标记通常属于不同上下文的语言任务不同，时间序列数据通常在单个时间点涉及重叠的上下文，例如每日、每周和季节性模式同时发生。这种重叠使得难以将时间序列模式有效地表示为标记，这给基于变压器的模型在充分捕获相关时间结构方面带来了挑战。

作为一个 TSPM（具体含义需结合上下文），一个模型必须具备哪些能力，又必须克服哪些挑战？

![20250326154842](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250326154842.png)

在讨论 TSPM 的设计之前，我们首先重新考虑时间序列是如何从以各种尺度采样的连续现实世界过程中生成的。例如，每日数据捕捉每小时的波动，而年度数据反映长期趋势和季节周期。这种多尺度、多周期性的性质给模型设计带来了重大挑战，因为每个尺度都强调不同的时间动态，必须有效地捕捉这些动态。图 1 说明了构建通用 TSPM 中的这一挑战。具体而言，较低的 CKA 相似性（Kornblith 等人，2019）表明跨层的表示更加多样化，这对于需要捕捉不规则模式和处理缺失数据的插补和异常检测等任务是有利的。在这些情况下，跨层的多样化表示有助于管理跨尺度和周期性的变化。相反，预测和分类任务受益于较高的 CKA 相似性，其中跨层的一致表示更好地捕捉稳定趋势和周期性模式。这种对比强调了设计一个通用模型的挑战，该模型要足够灵活以适应各种分析任务中的多尺度和多周期性模式，这些模式可能倾向于多样化或一致的表示。

## 二、问题剖析与解决策略

为了解决上述问题和挑战，我们提出了 TIMEMIXER++，这是一种通用的时间序列模式挖掘（TSPM）方法，旨在通过应对多尺度和多周期性动态的复杂性来捕获通用的、任务自适应的时间序列模式。其关键思想是同时在时域的多个尺度和频域的各种分辨率下捕获复杂的时间序列模式。具体而言，TIMEMIXER++通过（1）多分辨率时间成像（MRTI）、（2）时间图像分解（TID）、（3）多尺度混合（MCM）和（4）多分辨率混合（MRM）来处理多尺度时间序列，以揭示全面的模式。MRTI 将多尺度时间序列转换为多分辨率时间图像，从而能够在时域和频域中进行模式提取。TID 在潜在空间中应用双轴注意力来解开季节性和趋势模式，而 MCM 则在多个尺度上分层聚合这些模式。最后，MRM 自适应地整合了不同分辨率下的所有表示。如图 1 所示，TIMEMIXER++在 8 个分析任务中实现了最先进的性能，优于通用模型和特定任务模型。它的适应性体现在不同任务下的 CKA 相似性得分各不相同，这表明它比其他模型更能有效地捕捉不同的特定任务模式。我们的贡献总结如下：

### 2.1 解决方法

#### 2.1.1 Input Projection

![20250326202517](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250326202517.png)

先前的研究（2023；2023）采用通道独立策略，以避免将多个变量投影到难以区分的通道中（Liu等人，2024）。 相比之下，我们采用通道混合的方式来捕捉变量间的相互作用，这对于揭示时间序列数据中的全面模式至关重要。输入投影包含两个部分：通道混合和嵌入。

我们首先对最粗尺度$\mathbf{x}_M \in \mathbb{R}^{\lfloor\frac{T}{2^M}\rfloor\times C}$的变量维度应用自注意力机制，因为它保留了最全局的上下文信息，有助于更有效地整合跨变量的信息。其公式如下：
$$\mathbf{x}_M = \text{Channel-Attn}(\mathbf{Q}_M, \mathbf{K}_M, \mathbf{V}_M) \tag{2}$$
其中，Channel-Attn表示用于通道混合的变量级自注意力机制。查询、键和值$\mathbf{Q}_M, \mathbf{K}_M, \mathbf{V}_M \in \mathbb{R}^{C\times\lfloor\frac{T}{2^M}\rfloor}$ 是由$\mathbf{x}_M$的线性投影得到的。 然后，我们使用嵌入层将所有多尺度时间序列嵌入到一个深度模式集合$\mathcal{X}^0$中，可表示为$\mathcal{X}^0 = \{\mathbf{x}_0^0, \cdots, \mathbf{x}_M^0\} = \text{Embed}(\mathcal{X}_{init})$，其中$\mathbf{x}_m^0 \in \mathbb{R}^{\lfloor\frac{T}{2^m}\rfloor\times d_{\text{model}}}$，$d_{\text{model}}$表示深度模式的维度。

#### 2.1.2 MixerBlocks

我们以残差连接的方式堆叠混合块。对于第$(l + 1)$个块，输入是多尺度表示集合$\mathcal{X}^l$，前向传播过程可以形式化为：
$$\mathcal{X}^{l + 1} = \text{LayerNorm}(\mathcal{X}^l + \text{MixerBlock}(\mathcal{X}^l)) \tag{4}$$
其中，LayerNorm（层归一化）对不同尺度的模式进行归一化处理，有助于稳定训练过程。时间序列具有复杂的多尺度和多周期动态特性。多分辨率分析（Hartik ，1993）将时间序列在频域中建模为各种周期成分的组合。我们引入多分辨率时间成像技术，基于频域分析将一维多尺度时间序列转换为二维图像，同时保留原始时间序列的时间和频率域特征。这使得卷积方法能够有效地用于提取时间模式，并增强不同任务之间的通用性。具体而言，我们使用以下步骤处理多尺度时间序列：（1）多分辨率时间成像（MRTI）；（2）时间图像分解（TID）；（3）多尺度混合（MCM）；（4）多分辨率混合（MRM），以挖掘全面的时间序列模式。

##### 多分辨率时间成像

![20250326171745](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250326171745.png)

在每个混合块的开始，我们通过频域分析（Wu等人，2023）将输入$\mathcal{X}^l$转换为$(M + 1)×K$个多分辨率时间图像。为了捕捉具有代表性的周期模式，我们首先从最粗尺度$\mathbf{x}_M^l$中识别周期，这有助于实现全局交互。具体来说，我们对$\mathbf{x}_M^l$应用快速傅里叶变换（FFT），并选择幅度最高的前$K$个频率：
$$\mathbf{A}, \{f_1, \cdots, f_K\}, \{p_1, \cdots, p_K\} = \text{FFT}(\mathbf{x}_M^l) \tag{5}$$
其中$\mathbf{A} = \{A_{f_1}, \cdots, A_{f_{K}}\}$表示未归一化的幅度，$\{f_1, \cdots, f_K\}$是前$K$个频率，$p_k = \left\lceil \frac{T}{f_k} \right\rceil$，$k \in \{1, \cdots, K\}$表示相应的周期长度。然后，每个时间序列表示$\mathbf{x}_m^l$沿时间维度进行如下重塑：
$$
\begin{align*}
\text{MRTI}(\mathcal{X}^l) &= \{\mathcal{Z}_m^l\}_{m = 0}^M = \{\mathbf{z}_m^{(l,k)} \mid m = 0, \cdots, M; k = 1, \cdots, K\} \\
&= \left\{\underset{1D \to 2D}{\text{Reshape}_{m,k}}(\text{Padding}_{\sigma_{m,k}}(\mathbf{x}_m^l)) \mid m = 0, \cdots, M; k = 1, \cdots, K \right\}
\end{align*} \tag{6}
$$
其中$\text{Padding}_{\sigma_{m,k}}(\cdot)$对时间序列进行零填充，使其长度为$p_k \cdot \left\lceil \frac{\lfloor \frac{T}{f_k} \rfloor}{p_k} \right\rceil$，$\underset{1D \to 2D}{\text{Reshape}_{m,k}}(\cdot)$将其转换为一个$p_k \times \left\lceil \frac{\lfloor \frac{T}{f_k} \rfloor}{p_k} \right\rceil$的图像，记为$\mathbf{z}_m^{(l,k)}$。这里，$p_k$表示行数（周期长度），列数记为$f_{m,k} = \left\lceil \frac{\lfloor \frac{T}{f_k} \rfloor}{p_k} \right\rceil$，代表尺度$m$对应的频率。

##### 时间图像分解

![20250326203430](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250326203430.png)

时间序列模式本质上是嵌套的，具有重叠的尺度和周期。例如，每周销售数据既反映了日常购物习惯，又体现了更广泛的季节性趋势。传统方法（Wu等人，2021；Wang等人，2024b）对整个序列进行移动平均，往往会模糊不同的模式。为了解决这个问题，我们使用多分辨率时间图像，其中每个图像$\mathbf{z}_m^{(l,k)} \in \mathbb{R}^{p_k \times f_{m,k} \times d_{\text{model}}}$对特定的尺度和周期进行编码，从而能够更精细地分离季节性和趋势成分。通过对这些图像应用二维卷积，我们可以捕捉长程模式，并增强时间依赖关系的提取。每个图像中的列对应一个周期内的时间序列片段，而行表示不同周期内的一致时间点，这有助于双轴注意力机制：列轴注意力（$\text{Attention}_{\text{col}}$）捕捉周期内的季节性，行轴注意力（$\text{Attention}_{\text{row}}$）提取跨周期的趋势。每个轴特定的注意力集中在一个轴上，通过将非目标轴转置到批量维度来保持计算效率。对于列轴注意力，查询、键和值$\mathbf{Q}_{\text{col}}, \mathbf{K}_{\text{col}}, \mathbf{V}_{\text{col}} \in \mathbb{R}^{f_{m,k} \times d_{\text{model}}}$通过二维卷积计算，这些值在所有图像中共享，行轴注意力的计算方式类似，分别得到$\mathbf{Q}_{\text{row}}, \mathbf{K}_{\text{row}}, \mathbf{V}_{\text{row}}$。然后，季节性和趋势成分计算如下：
$$\mathbf{s}_m^{(l,k)} = \text{Attention}_{\text{col}}(\mathbf{Q}_{\text{col}}, \mathbf{K}_{\text{col}}, \mathbf{V}_{\text{col}}), \quad \mathbf{t}_m^{(l,k)} = \text{Attention}_{\text{row}}(\mathbf{Q}_{\text{row}}, \mathbf{K}_{\text{row}}, \mathbf{V}_{\text{row}}) \tag{7}$$
其中$\mathbf{s}_m^{(l,k)}, \mathbf{t}_m^{(l,k)} \in \mathbb{R}^{p_k \times f_{m,k} \times d_{\text{model}}}$分别表示季节性图像和趋势图像。这里，在注意力计算后，转置的轴被恢复以还原原始图像大小。

##### 多尺度混合

![20250326203605](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250326203605.png)

对于每个周期$p_k$，我们得到$M + 1$个季节性时间图像和$M + 1$个趋势时间图像，分别记为$\{\mathbf{s}_m^{(l,k)}\}_{m = 0}^M$和$\{\mathbf{t}_m^{(l,k)}\}_{m = 0}^M$。二维结构使我们能够使用二维卷积层同时对季节性和趋势模式进行建模，这比传统线性层在捕捉长程依赖关系方面更高效（Wang等人，2024b）。对于多尺度季节性时间图像，较长的模式可以看作是较短模式的组合（例如，年度降雨量模式由每月变化形成）。因此，我们以残差连接的方式混合季节性模式，从细尺度到粗尺度进行。为了便于这种信息流，我们在第$m$个尺度上应用二维卷积层，形式化为：
$$\text{for } m: 1 \to M \text{ do: } \mathbf{s}_m^{(l,k)} = \mathbf{s}_m^{(l,k)} + \text{2D-Conv}(\mathbf{s}_{m - 1}^{(l,k)}) \tag{8}$$
其中2D - Conv由两个时间步长为2的二维卷积层组成。与季节性模式不同，对于多尺度趋势时间图像，较粗的尺度自然地突出了整体趋势。因此，我们采用从粗到细的混合策略，并在第$m$个尺度上应用二维转置卷积层，形式化为：
$$\text{for } m: M - 1 \to 0 \text{ do: } \mathbf{t}_m^{(l,k)} = \mathbf{t}_m^{(l,k)} + \text{2D-TransConv}(\mathbf{t}_{m + 1}^{(l,k)}) \tag{9}$$
其中2D - TransConv由两个时间步长为2的二维转置卷积层组成。混合后，季节性和趋势模式通过求和进行聚合，并重新转换回一维结构，如下所示：
$$\mathbf{z}_m^{(l,k)} = \underset{2D \to 1D}{\text{Reshape}_{m,k}}(\mathbf{s}_m^{(l,k)} + \mathbf{t}_m^{(l,k)}), \quad m \in \{0, \cdots, M\} \tag{10}$$
其中$\underset{2D \to 1D}{\text{Reshape}_{m,k}}(\cdot)$将一个$p_k \times f_{m,k}$的图像转换为长度为$p_k \cdot f_{m,k}$的时间序列。

#### 多分辨率混合

![20250326203835](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250326203835.png)

最后，在每个尺度上，我们自适应地混合$K$个周期。幅度$\mathbf{A}$捕捉每个周期的重要性，我们按如下方式聚合模式$\{\mathbf{z}_m^{(l,k)}\}_{k = 1}^K$：
$$\{\hat{\mathbf{A}}_{f_k}\}_{k = 1}^K = \text{Softmax}(\{\mathbf{A}_{f_k}\}_{k = 1}^K), \quad \mathbf{x}_m^l = \sum_{k = 1}^K \hat{\mathbf{A}}_{f_k} \circ \mathbf{z}_m^{(l,k)}, \quad m \in \{0, \cdots, M\} \tag{11}$$
其中Softmax对权重进行归一化，$\circ$表示元素级乘法。

#### 2.1.3 Output Projection

经过$L$个MixerBlock堆叠后，我们得到多尺度表示集合$\mathcal{X}^L$。正如第1节所讨论的，不同尺度捕捉到不同的时间模式，且不同任务的需求也各不相同。因此，我们建议使用多个预测头，每个预测头专门针对一个特定尺度，并将它们的输出进行集成。这种设计是任务自适应的，使得每个预测头能够专注于其对应尺度下的相关特征，而集成操作则聚合互补信息，以增强预测的稳健性。
$$\text{output} = \text{Ensemble}(\{\text{Head}_m(\mathbf{x}_m^L)\}_{m = 0}^M) \tag{3}$$
其中，$\text{Ensemble}(\cdot)$表示集成方法（例如，求平均值或加权求和），$\text{Head}_m(\cdot)$是第$m$个尺度的预测头，通常是一个线性层。

### 2.2 模型结构

![20250326232907](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250326232907.png)

## 三、实验验证与结果分析

### 3.1 Main Result

#### 长期预测

![20250330155448](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330155448.png)

#### 短期预测

![20250330155333](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330155333.png)

#### 单变量短期预测

![20250330155526](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330155526.png)

#### 插补

![20250330155615](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330155615.png)

#### 少样本

![20250330155630](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330155630.png)

#### 零样本

![20250330155651](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330155651.png)

#### 分类和异常检测

![20250330155828](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330155828.png)

### 3.2 消融实验

![20250330160059](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330160059.png)

![20250330160254](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330160254.png)

### 3.3 可视化

![20250330160446](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330160446.png)
