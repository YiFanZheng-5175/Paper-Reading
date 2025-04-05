# Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting

>领域：时间序列预测  
>发表在：2021 NeurIPS  
>模型名字：Autoformer  
>文章链接：[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://arxiv.org/abs/2106.13008)  
>代码仓库：[Time-Series-Library](https://github.com/thuml/Time-Series-Library)  
![20250330205309](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330205309.png)

## 一、研究背景与问题提出

### 1.1 研究现状

以前基于 Transformer 的预测模型 [48,23,26] 主要专注于将自注意力改进为稀疏版本。虽然性能显著提高，但这些模型仍然使用点式表示聚合。因此，在提高效率的过程中，由于稀疏的点式连接，它们会牺牲信息利用率，从而导致时间序列长期预测的瓶颈。

## 二、问题剖析与解决策略

为了推理复杂的时间模式，我们尝试采用分解的思想，这是时间序列分析中的一种标准方法 [1,33]。它可用于处理复杂的时间序列并提取更具可预测性的成分。然而，在预测背景下，由于未来是未知的，它只能用作对过去序列的预处理 [20]。这种常见用法限制了分解的能力，并忽略了分解后的成分之间潜在的未来相互作用。因此，我们试图超越分解的预处理用法，并提出一种通用架构，以赋予深度预测模型具有渐进式分解的内在能力。此外，分解可以解开纠缠的时间模式，并突出时间序列的内在特性 [20]。

受益于此，我们尝试利用序列周期性来更新自注意力中的逐点连接。我们观察到周期中相同相位位置的子序列通常呈现出相似的时间过程。因此，我们尝试基于由序列周期性导出的过程相似性构建一个序列级连接。基于上述动机，我们提出了一种原始的 Autoformer 来替代 Transformer 用于长期时间序列预测。Autoformer 仍然遵循残差和编码器 - 解码器结构，但将 Transformer 革新为一种分解预测架构。通过将我们提出的分解块嵌入为内部算子，Autoformer 可以从预测的隐藏变量中逐步分离出长期趋势信息。这种设计允许我们的模型在预测过程中交替地分解和细化中间结果。

受随机过程理论 [9,30] 的启发，Autoformer 引入了一种自相关机制来替代自注意力，该机制基于序列周期性发现子序列相似性，并从底层周期中聚合相似的子序列。这种序列级机制对于长度为 L 的序列实现了O(LlogL)的复杂度，并通过将逐点表示聚合扩展到子序列级别打破了信息利用瓶颈。

### 2.1 解决方法

#### 2.1.1 分解架构

我们将Transformer改进为一种深度分解架构（图1），包括内部序列分解模块、自相关机制以及相应的编码器和解码器。

![20250330210109](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330210109.png)

- **序列分解模块**：为了在长期预测的背景下学习复杂的时间模式，我们采用了分解的思路，它可以将序列分离为趋势 - 循环部分和季节性部分。这两个部分分别反映了序列的长期变化趋势和季节性特征。然而，直接对未来序列进行分解是不可行的，因为未来是未知的。为了解决这个困境，我们提出将序列分解模块作为Autoformer的内部操作（图1），它可以从预测的中间隐藏变量中逐步提取长期平稳趋势。具体来说，我们使用移动平均来平滑周期性波动并突出长期趋势。对于长度为$L$的输入序列$\mathcal{X} \in \mathbb{R}^{L \times d}$，过程如下：
  - $\mathcal{X}_t = \text{AvgPool}(\text{Padding}(\mathcal{X}))$
  - $\mathcal{X}_s = \mathcal{X} - \mathcal{X}_t$
其中$\mathcal{X}_s, \mathcal{X}_t \in \mathbb{R}^{L \times d}$分别表示季节性部分和提取出的趋势 - 循环部分。我们采用$\text{AvgPool}(\cdot)$进行移动平均，并使用填充操作来保持序列长度不变。我们用$\mathcal{X}_s, \mathcal{X}_t = \text{SeriesDecomp}(\mathcal{X})$来概括上述方程，这是一个模型内部模块。
- **模型输入**：编码器部分的输入是过去$I$个时间步$\mathcal{X}_{\text{en}} \in \mathbb{R}^{I \times d}$。作为一种分解架构（图1），Autoformer解码器的输入包含需要细化的季节性部分$\mathcal{X}_{\text{des}} \in \mathbb{R}^{(\frac{I}{2}+O) \times d}$和趋势 - 循环部分$\mathcal{X}_{\text{det}} \in \mathbb{R}^{(\frac{I}{2}+O) \times d}$ 。每个初始化由两部分组成：从编码器输入$\mathcal{X}_{\text{en}}$后半部分分解出的长度为$\frac{I}{2}$的分量，用于提供近期信息；长度为$O$的由标量填充的占位符。其公式如下：
  - $\mathcal{X}_{\text{ens}}, \mathcal{X}_{\text{ent}} = \text{SeriesDecomp}(\mathcal{X}_{\text{en}_{\frac{I}{2}:I}})$
  - $\mathcal{X}_{\text{des}} = \text{Concat}(\mathcal{X}_{\text{ens}}, \mathcal{X}_0)$
  - $\mathcal{X}_{\text{det}} = \text{Concat}(\mathcal{X}_{\text{ent}}, \mathcal{X}_{\text{Mean}})$
其中$\mathcal{X}_{\text{ens}}, \mathcal{X}_{\text{ent}} \in \mathbb{R}^{\frac{I}{2} \times d}$分别表示$\mathcal{X}_{\text{en}}$的季节性部分和趋势 - 循环部分，$\mathcal{X}_0, \mathcal{X}_{\text{Mean}} \in \mathbb{R}^{O \times d}$分别表示用零填充的占位符和$\mathcal{X}_{\text{en}}$的均值。
- **编码器**：如图1所示，编码器专注于季节性部分的建模。编码器的输出包含过去的季节性信息，并将作为交叉信息来帮助解码器细化预测结果。假设我们有$N$个编码器层。第$l$个编码器层的总体方程概括为$\mathcal{X}_{\text{en}}^l = \text{Encoder}(\mathcal{X}_{\text{en}}^{l - 1})$。详细内容如下：
  - $\mathcal{S}_{\text{en}}^{l,1}, \_ = \text{SeriesDecomp}(\text{Auto-Correlation}(\mathcal{X}_{\text{en}}^{l - 1}) + \mathcal{X}_{\text{en}}^{l - 1})$
  - $\mathcal{S}_{\text{en}}^{l,2}, \_ = \text{SeriesDecomp}(\text{FeedForward}(\mathcal{S}_{\text{en}}^{l,1}) + \mathcal{S}_{\text{en}}^{l,1})$
其中“\_”是被消除的趋势部分。$\mathcal{X}_{\text{en}}^l = \mathcal{S}_{\text{en}}^{l,2}, l \in \{1, \dots, N\}$表示第$l$个编码器层的输出，$\mathcal{X}_{\text{en}}^0$是嵌入后的$\mathcal{X}_{\text{en}}$ 。$\mathcal{S}_{\text{en}}^{l,i}, i \in \{1, 2\}$分别表示第$l$层中第$i$个序列分解模块后的季节性分量。我们将在下一节详细描述自相关机制（$\text{Auto-Correlation}(\cdot)$），它可以无缝替换自注意力机制。
- **解码器**：解码器包含两个部分：用于累积结构前趋势 - 循环分量的部分，以及堆叠的用于累积季节性分量的自相关机制（图1）。每个解码器层包含内部自相关机制和编码器 - 解码器自相关机制，分别用于细化预测并利用过去的季节性信息。注意，模型在解码器中从中间隐藏变量中提取潜在趋势，这使得Autoformer能够逐步细化趋势预测，并消除自相关中基于周期的依赖关系发现的干扰信息。假设存在$M$个解码器层。对于第$l$个解码器层，利用来自编码器的隐藏变量$\mathcal{X}_{\text{en}}^N$ ，其方程可概括为$\mathcal{X}_{\text{de}}^l = \text{Decoder}(\mathcal{X}_{\text{de}}^{l - 1}, \mathcal{X}_{\text{en}}^N)$。解码器可形式化为如下：
  - $\mathcal{S}_{\text{de}}^{l,1}, \mathcal{T}_{\text{de}}^{l,1} = \text{SeriesDecomp}(\text{Auto-Correlation}(\mathcal{X}_{\text{de}}^{l - 1}) + \mathcal{X}_{\text{de}}^{l - 1})$
  - $\mathcal{S}_{\text{de}}^{l,2}, \mathcal{T}_{\text{de}}^{l,2} = \text{SeriesDecomp}(\text{Auto-Correlation}(\mathcal{S}_{\text{de}}^{l,1}, \mathcal{X}_{\text{en}}^N) + \mathcal{S}_{\text{de}}^{l,1})$
  - $\mathcal{S}_{\text{de}}^{l,3}, \mathcal{T}_{\text{de}}^{l,3} = \text{SeriesDecomp}(\text{FeedForward}(\mathcal{S}_{\text{de}}^{l,2}) + \mathcal{S}_{\text{de}}^{l,2})$
  - $\mathcal{T}_{\text{de}}^l = \mathcal{T}_{\text{de}}^{l - 1} + \mathcal{W}_{l,1} * \mathcal{T}_{\text{de}}^{l,1} + \mathcal{W}_{l,2} * \mathcal{T}_{\text{de}}^{l,2} + \mathcal{W}_{l,3} * \mathcal{T}_{\text{de}}^{l,3}$
其中$\mathcal{X}_{\text{de}}^l = \mathcal{S}_{\text{de}}^{l,3}, l \in \{1, \dots, M\}$表示第$l$个解码器层的输出。$\mathcal{X}_{\text{de}}^0$是从$\mathcal{X}_{\text{des}}$嵌入得到的用于深度变换的部分，$\mathcal{T}_{\text{de}}^0 = \mathcal{X}_{\text{det}}$用于累积。$\mathcal{S}_{\text{de}}^{l,i}, \mathcal{T}_{\text{de}}^{l,i}, i \in \{1, 2, 3\}$分别表示第$l$层中第$i$个序列分解模块后的季节性分量和趋势 - 循环分量。$\mathcal{W}_{l,i}, i \in \{1, 2, 3\}$表示第$i$个提取趋势$\mathcal{T}_{\text{de}}^{l,i}$的投影器。

最终预测结果是两个细化后的分解分量之和，即$\mathcal{W}_s * \mathcal{X}_{\text{de}}^M + \mathcal{T}_{\text{de}}^M$，其中$\mathcal{W}_s$用于将深度变换后的季节性分量$\mathcal{X}_{\text{de}}^M$投影到目标维度。

#### 2.1.2 自相关机制

![20250330221515](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330221515.png)

![20250330221744](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330221744.png)

如图2所示，我们提出带有序列级连接的自相关机制，以扩展信息利用率。自相关机制通过计算序列自相关来发现基于周期的依赖关系，并通过时间延迟聚合来聚合相似的子序列。

- **基于周期的依赖关系**：据观察，周期中相同的相位位置自然会提供相似的子过程。受随机过程理论启发，对于一个真实离散时间过程$\{\mathcal{X}_t\}$，我们可以通过以下方程得到自相关函数$\mathcal{R}_{\mathcal{X}\mathcal{X}}(\tau)$：
  - $\mathcal{R}_{\mathcal{X}\mathcal{X}}(\tau) = \lim_{L \to \infty} \frac{1}{L} \sum_{t = 1}^{L} \mathcal{X}_t \mathcal{X}_{t - \tau}$
$\mathcal{R}_{\mathcal{X}\mathcal{X}}(\tau)$反映了$\{\mathcal{X}_t\}$与其$\tau$延迟序列$\{\mathcal{X}_{t - \tau}\}$之间的时间延迟相似性。如图2所示，我们将自相关函数$\mathcal{R}(\tau)$用作估计周期长度$\tau$的未归一化置信度。然后，我们选择最有可能的$k$个周期长度$\tau_1, \dots, \tau_k$ 。基于周期的依赖关系由上述估计的周期推导得出，并可以通过相应的自相关进行加权。
- **时间延迟聚合**：基于周期的依赖关系连接了估计周期的相应子序列。因此，我们提出时间延迟聚合模块（图2），它可以根据选定的时间延迟$\tau_1, \dots, \tau_k$ 滚动序列。此操作可以对齐处于估计周期相同相位位置的相似子序列，这与自注意力族中的逐点乘积聚合不同。最后，我们通过softmax归一化置信度来聚合子序列。对于单头情况，我们有长度为$L$的查询序列$\mathcal{X}$，通过投影器得到查询$Q$、键$K$和值$V$ 。因此，它可以无缝替换自注意力。自相关机制为：
  - $\tau_1, \dots, \tau_k = \text{arg Top}_k(\mathcal{R}_{Q,K}(\tau))$，其中$\tau \in \{1, \dots, L\}$
  - $\widetilde{\mathcal{R}}_{Q,K}(\tau_1), \dots, \widetilde{\mathcal{R}}_{Q,K}(\tau_k) = \text{SoftMax}(\mathcal{R}_{Q,K}(\tau_1), \dots, \mathcal{R}_{Q,K}(\tau_k))$
  - $\text{Auto-Correlation}(Q, K, V) = \sum_{i = 1}^{k} \text{Roll}(V, \tau_i) \widetilde{\mathcal{R}}_{Q,K}(\tau_i)$
其中$\text{arg Top}_k(\cdot)$是获取Top - $k$自相关的参数，且$k = \lfloor c \times \log L \rfloor$ ，$c$是一个超参数。$\mathcal{R}_{Q,K}$是序列$Q$和$K$之间的自相关。$\text{Roll}(\mathcal{X}, \tau)$表示对$\mathcal{X}$进行时间延迟$\tau$的操作，在此过程中，超出第一个位置的元素会重新引入到最后一个位置。对于编码器 - 解码器自相关（图1），$\mathcal{K}, \mathcal{V}$来自编码器$\mathcal{X}_{\text{en}}^N$ ，$Q$来自解码器的前一个模块。

对于Autoformer中使用的多头版本，具有$d_{\text{model}}$通道的隐藏变量，$h$个头，第$i$个头的查询、键和值为$Q_i, K_i, V_i \in \mathbb{R}^{L \times \frac{d_{\text{model}}}{h}}$，$i \in \{1, \dots, h\}$ 。过程如下：
    - $\text{MultiHead}(Q, K, V) = \mathcal{W}_{\text{output}} * \text{Concat}(\text{head}_1, \dots, \text{head}_h)$
    - 其中$\text{head}_i = \text{Auto-Correlation}(Q_i, K_i, V_i)$

- **高效计算**：对于基于周期的依赖关系，这些依赖关系指向潜在周期相同相位位置的子过程，并且本质上是稀疏的。在这里，我们选择最有可能的延迟，以避免选择相反的相位。因为我们聚合$O(\log L)$个长度为$L$的序列，所以方程(6)和(7)的复杂度为$O(L \log L)$ 。对于自相关计算（方程(5)），给定时间序列$\{\mathcal{X}_t\}$，根据维纳 - 辛钦定理，可以通过快速傅里叶变换（FFT）计算$\mathcal{R}_{\mathcal{X}\mathcal{X}}(\tau)$：
  - $\mathcal{S}_{\mathcal{X}\mathcal{X}}(f) = \mathcal{F}(\mathcal{X}_t) \mathcal{F}^*(\mathcal{X}_t) = \int_{-\infty}^{\infty} \mathcal{X}_t e^{-i 2 \pi f t} \mathrm{d}t \overline{\int_{-\infty}^{\infty} \mathcal{X}_t e^{-i 2 \pi f t} \mathrm{d}t}$
  - $\mathcal{R}_{\mathcal{X}\mathcal{X}}(\tau) = \mathcal{F}^{-1}(\mathcal{S}_{\mathcal{X}\mathcal{X}}(f)) = \int_{-\infty}^{\infty} \mathcal{S}_{\mathcal{X}\mathcal{X}}(f) e^{i 2 \pi f \tau} \mathrm{d}f$
其中$\tau \in \{1, \dots, L\}$ ，$\mathcal{F}$表示FFT，$\mathcal{F}^{-1}$是其逆变换。$*$表示共轭运算，$\mathcal{S}_{\mathcal{X}\mathcal{X}}(f)$在频域中。注意，通过FFT可以一次计算所有延迟$\{1, \dots, L\}$的序列自相关。因此，自相关实现了$O(L \log L)$的复杂度。
- **自相关与自注意力族对比**：与逐点自注意力族不同，自相关呈现出序列级连接（图3）。具体来说，对于时间依赖关系，我们基于周期性找到子序列之间的依赖关系。相比之下，自注意力族仅计算离散点之间的关系。尽管一些自注意力（文献[26, 48]）考虑了局部信息，但它们仅利用这些信息来帮助逐点依赖关系发现。对于信息聚合，我们采用时间延迟模块来聚合来自潜在周期的相似子序列。相比之下，自注意力通过点积聚合选定的点。得益于内在的稀疏性和子序列级的表示聚合，自相关可以同时提高计算效率和信息利用率。

### 2.2 模型结构

## 三、实验验证与结果分析

### 3.1 可视化实验

#### 时间序列分解

![20250330222558](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330222558.png)
如图4所示，若没有我们的序列分解模块，预测模型无法捕捉到趋势增长以及季节性部分的峰值。通过添加序列分解模块，Autoformer能够逐步聚合和细化序列中的趋势 - 循环部分。这一设计也有助于对季节性部分的学习，特别是峰值和谷值的捕捉。这验证了我们所提出的渐进式分解架构的必要性。

#### 依赖关系学习

![20250330222639](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330222639.png)

图5(a)中标记的时间延迟大小表明了最可能的周期。我们所学习到的周期性能够引导模型通过$\text{Roll}(\mathcal{X}, \tau_i)$ （$i \in \{1, \dots, 6\}$）聚合来自相同或相邻周期相位的子序列。对于最后一个时间步（下降阶段），与自注意力机制相比，自相关机制能够充分利用所有相似子序列，没有遗漏或错误。这验证了Autoformer能够更充分、精确地发现相关信息。

#### 复杂季节性建模

![20250330222815](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250330222815.png)

如图6所示，Autoformer从深度表示中学习到的延迟能够反映原始序列的真实季节性。例如，对于按日记录的Exchange数据集，学习到的延迟呈现出月度、季度和年度周期（图6(b)）。对于按小时记录的Traffic数据集（图6(c)），学习到的延迟显示出24小时和168小时的间隔，这与现实场景中的每日和每周周期相匹配。这些结果表明，Autoformer能够从深度表示中捕捉现实世界序列的复杂季节性，并进一步提供可被人类理解的预测。
