# Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting

>领域：时间序列预测  
>发表在：NeurIPS 2022  
>模型名字：Non-stationary Transformers  
>文章链接：[Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting](https://arxiv.org/abs/2205.14415)  
>代码仓库：[https://github.com/thuml/Nonstationary_Transformers](https://github.com/thuml/Nonstationary_Transformers)  
![20250423222116](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250423222116.png)

## 一、研究背景与问题提出

### 1.1 研究现状

尽管 Transformer 有着卓越的架构设计，但由于数据的非平稳性，它们在预测实际时间序列时仍面临挑战。非平稳时间序列的特点是统计属性和联合分布随时间不断变化，这使得时间序列的可预测性降低  [6, 16] 。此外，让深度模型在变化的分布上具有良好的泛化能力是一个根本问题  [28, 21, 5] 。在以往的工作中，通常认可通过平稳化对时间序列进行预处理  [26, 29, 17] ，这可以减弱原始时间序列的非平稳性，以提高可预测性，并为深度模型提供更稳定的数据分布。

尽管平稳性对时间序列的可预测性很重要 [6, 16] ，但实际中的时间序列往往呈现非平稳性。为解决这一问题，经典统计方法自回归积分滑动平均模型（ARIMA） [7, 8] 通过差分使时间序列平稳化。对于深度模型而言，由于非平稳性带来的分布变化问题使深度预测更难实现，平稳化方法得到了广泛探索，且常被用作深度模型输入的预处理。自适应归一化（Adaptive Norm） [26] 根据采样集的全局统计量对每个序列片段应用z - 分数归一化。DAIN [29] 使用非线性神经网络根据观测到的训练分布对时间序列进行自适应平稳化。RevIN [17] 引入了两阶段实例归一化 [33] ，分别对模型输入和输出进行变换以减小每个序列的差异。相比之下，我们发现直接对时间序列进行平稳化会损害模型对特定时间依赖性进行建模的能力。因此，与以往方法不同，非平稳Transformer除了进行平稳化之外，还进一步开发了去平稳注意力，以将原始序列的固有非平稳性重新纳入考量。

![20250423222538](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250423222538.png)
>图1：针对具有不同均值$\mu$和标准差$\sigma$的不同序列所学习到的时间注意力可视化。(a) 来自在原始序列上训练的普通Transformer  [34] 。(b) 来自在平稳化序列上训练的Transformer，其呈现出相似的注意力。(c) 来自非平稳Transformer，其引入了去平稳注意力以避免过度平稳化。

非平稳性是实际时间序列的固有属性，也是发现用于预测的时间依赖性的良好指引。通过实验我们观察到，在平稳化序列上进行训练会削弱 Transformer 学习到的注意力的区分度。虽然普通 Transformer  [34] 能够从不同序列中捕捉到独特的时间依赖性（如图 1(a) 所示），但在平稳化序列上训练的 Transformer 会产生难以区分的注意力（如图 1(b) 所示）。这个被称为过度平稳化的问题会带来意外的副作用，使 Transformer 无法捕捉重要的时间依赖性，限制模型的预测能力，甚至导致模型生成与真实值存在巨大非平稳偏差的输出。因此，如何在减弱时间序列非平稳性以提高可预测性的同时，缓解过度平稳化问题以提升模型能力，是进一步提高预测性能的关键问题。

## 二、问题剖析与解决策略

在本文中，我们探究了平稳化在时间序列预测中的作用，并提出了非平稳 Transformer（Non - stationary Transformers）作为通用框架，赋予 Transformer  [34] 及其高效变体  [19, 39, 37] 对实际时间序列强大的预测能力。所提出的框架包含两个相互关联的模块：序列平稳化（Series Stationarization）以提高非平稳序列的可预测性，以及去平稳注意力（De - stationary Attention）以缓解过度平稳化。从技术上讲，序列平稳化采用一种简单而有效的归一化策略，在不增加额外参数的情况下统一每个序列的关键统计量。而去平稳注意力则近似未平稳化数据的注意力，并补偿原始序列的固有非平稳性。受益于上述设计，非平稳 Transformer 能够利用平稳化序列的高可预测性以及从原始非平稳数据中发现的关键时间依赖性。我们的方法在六个实际基准数据集上取得了领先的性能，并且可以推广到各种 Transformer 模型以进一步改进。

### 2.1 解决方法

非平稳时间序列让深度模型难以完成预测任务，因为在推断过程中，模型很难在统计量发生变化（典型的如均值和标准差不同）的序列上有良好的泛化表现。前期工作RevIN [17] 对每个输入应用带可学习仿射参数的实例归一化，并将统计量还原到对应的输出，这使得每个序列都遵循相似的分布。我们通过实验发现，这种设计在没有可学习参数的情况下也能很好地发挥作用。因此，我们提出一种更直接且有效的设计，将 Transformer 作为基础模型，无需额外参数，我们称之为序列平稳化。如图2所示，它包含两个相应的操作：首先是归一化模块，用于处理由不同均值和标准差导致的非平稳序列；最后是反归一化模块，将模型输出还原为原始统计量。以下是详细内容。

![20250423222116](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250423222116.png)

#### 2.1.1 序列平稳化

**归一化模块** 为了减弱每个输入序列的非平稳性，我们通过时间上的滑动窗口进行归一化操作。对于每个输入序列 $\mathbf{x} = [x_1, x_2, \ldots, x_S]^{\top} \in \mathbb{R}^{S \times C}$ ，我们进行平移和缩放操作，得到 $\mathbf{x}' = [x_1', x_2', \ldots, x_S']^{\top} \in \mathbb{R}^{S \times C}$ ，其中 $S$ 和 $C$ 分别表示序列长度和变量数量。归一化模块可以表示为：
$$
\mu_{\mathbf{x}} = \frac{1}{S} \sum_{i = 1}^{S} x_i, \quad \sigma_{\mathbf{x}}^2 = \frac{1}{S} \sum_{i = 1}^{S} (x_i - \mu_{\mathbf{x}})^2, \quad x_i' = \frac{1}{\sigma_{\mathbf{x}}} \odot (x_i - \mu_{\mathbf{x}})
\tag{1}
$$
其中 $\mu_{\mathbf{x}}, \sigma_{\mathbf{x}} \in \mathbb{R}^{C \times 1}$ ，$\frac{1}{\sigma_{\mathbf{x}}}$ 表示元素级除法，$\odot$ 是元素级乘法。注意，归一化模块减小了每个输入时间序列之间的分布差异，使模型输入的分布更稳定。

**反归一化模块** 如图2所示，在基础模型 $\mathcal{H}$ 预测出长度为 $O$ 的未来值后，我们采用反归一化操作，利用 $\sigma_{\mathbf{x}}$ 和 $\mu_{\mathbf{x}}$ 对模型输出 $\mathbf{y}' = [y_1', y_2', \ldots, y_O']^{\top} \in \mathbb{R}^{O \times C}$ 进行变换，得到最终预测结果 $\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, \ldots, \hat{y}_O]^{\top}$ 。反归一化模块可以表示为：
$$
\mathbf{y}' = \mathcal{H}(\mathbf{x}'), \quad \hat{y}_i = \sigma_{\mathbf{x}} \odot y_i' + \mu_{\mathbf{x}}
\tag{2}
$$

通过两阶段变换，基础模型将接收到平稳化后的输入，这些输入遵循稳定的分布，也更容易泛化。这也使得模型对时间序列的平移和缩放扰动具有等变性，从而有利于实际序列的预测。

#### 2.1.2 去平稳注意力

虽然每个时间序列的统计量被明确还原到对应的预测中，但仅通过反归一化无法完全恢复原始序列的非平稳性。例如，序列平稳化可能会从不同的时间序列 $\mathbf{x}_1, \mathbf{x}_2$ （即 $\mathbf{x}_2 = \alpha\mathbf{x}_1 + \beta$ ）生成相同的平稳化输入 $\mathbf{x}'$ ，并且基础模型会得到相同的注意力，而这种注意力无法捕捉到关键的时间依赖性（如图1所示）。换句话说，过度平稳化带来的不良影响发生在深度模型内部，尤其是在注意力计算过程中。此外，非平稳时间序列在平稳化之前被分割并归一化为几个序列块，这些序列块具有相同的均值和方差，更符合原始数据的分布。因此，模型更有可能生成过度平稳且缺乏有效信息的输出，这与原始序列的自然非平稳性是相悖的。

为了解决序列平稳化带来的过度平稳化问题，我们提出一种新颖的去平稳注意力机制，它可以近似从原始非平稳数据中获得的注意力，从而恢复固有非平稳性信息并发现特定的时间依赖性。

**简单模型分析** 如前所述，过度平稳化问题是由固有非平稳性信息的消失导致的，这会使基础模型无法捕捉到用于预测的关键时间依赖性。因此，我们尝试近似从原始非平稳序列中学习到的注意力。我们从自注意力公式 [34] 开始：
$$
\mathrm{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}}\right) \mathbf{V}
\tag{3}
$$
其中 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{S \times d_k}$ 分别是长度为 $S$ 的查询、键和值向量，维度为 $d_k$ ，Softmax 操作按行进行。为了简化分析，我们假设嵌入层和前馈层 $f$ 具有线性性质，并且 $f$ 是在时间点上分别进行计算的，即查询向量 $\mathbf{Q} = [q_1, q_2, \ldots, q_S]^{\top}$ 中的每个 $q_i$ 都可以根据输入序列 $\mathbf{x} = [x_1, x_2, \ldots, x_S]^{\top}$ 计算为 $q_i = f(x_i)$ 。由于习惯上对每个时间序列变量进行归一化，并且假设它在尺度上占主导地位，我们进一步假设序列 $\mathbf{x}$ 的每个变量具有相同的方差，因此原始的 $\sigma_{\mathbf{x}} \in \mathbb{R}^{C \times 1}$ 简化为一个标量。

归一化后，模型接收平稳化后的输入 $\mathbf{x}' = (\mathbf{x} - \mathbf{1}\mu_{\mathbf{x}}) / \sigma_{\mathbf{x}}$ ，其中 $\mathbf{1} \in \mathbb{R}^{S \times 1}$ 是全一向量。基于线性性质假设，可以证明注意力层将接收 $\mathbf{Q}' = [f(x_1'), \ldots, f(x_S')]^{\top} = (\mathbf{Q} - \mathbf{1}\mu_{\mathbf{Q}}) / \sigma_{\mathbf{x}}$ ，其中 $\mu_{\mathbf{Q}} \in \mathbb{R}^{d_k \times 1}$ 是 $\mathbf{Q}$ 在时间维度上的均值（详细证明见附录A）。$\mathbf{K}$ 也有相应的变换。因此，自注意力中的 Softmax 输入应该是 $\mathbf{Q}\mathbf{K}^{\top} / \sqrt{d_k}$ ，而现在的注意力是基于 $\mathbf{Q}', \mathbf{K}'$ 计算的：
$$
\begin{align*}
\mathbf{Q}'\mathbf{K}'^{\top} &= \frac{1}{\sigma_{\mathbf{x}}^2} \left( \mathbf{Q}\mathbf{K}^{\top} - \mathbf{1}(\mu_{\mathbf{Q}}^{\top}\mathbf{K}^{\top}) - (\mathbf{Q}\mu_{\mathbf{K}})\mathbf{1}^{\top} + \mathbf{1}(\mu_{\mathbf{Q}}^{\top}\mu_{\mathbf{K}})\mathbf{1}^{\top} \right) \\
\mathrm{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}} \right) &= \mathrm{Softmax}\left( \frac{\sigma_{\mathbf{x}}^2 \mathbf{Q}'\mathbf{K}'^{\top} + \mathbf{1}(\mu_{\mathbf{Q}}^{\top}\mathbf{K}^{\top}) + (\mathbf{Q}\mu_{\mathbf{K}})\mathbf{1}^{\top} - \mathbf{1}(\mu_{\mathbf{Q}}^{\top}\mu_{\mathbf{K}})\mathbf{1}^{\top}}{\sqrt{d_k}} \right)
\tag{4}
\end{align*}
$$

我们发现 $\mathbf{Q}\mu_{\mathbf{K}} \in \mathbb{R}^{S \times 1}$ 且 $\mu_{\mathbf{Q}}^{\top}\mu_{\mathbf{K}} \in \mathbb{R}$ ，它们分别对 $\sigma_{\mathbf{x}}^2 \mathbf{Q}'\mathbf{K}'^{\top} \in \mathbb{R}^{S \times S}$ 的每一列和每个元素进行重复操作。由于 Softmax 对输入在行维度上的相同平移是不变的，我们有以下等式：
$$
\mathrm{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_k}} \right) = \mathrm{Softmax}\left( \frac{\sigma_{\mathbf{x}}^2 \mathbf{Q}'\mathbf{K}'^{\top} + \mathbf{1}\mu_{\mathbf{Q}}^{\top}\mathbf{K}^{\top}}{\sqrt{d_k}} \right)
\tag{5}
$$

公式(5)推导出了从原始序列 $\mathbf{x}$ 学习到的注意力 $\mathrm{Softmax}(\mathbf{Q}\mathbf{K}^{\top} / \sqrt{d_k})$ 的直接表达式。除了来自平稳化序列 $\mathbf{x}'$ 的当前 $\mathbf{Q}', \mathbf{K}'$ ，这个表达式还需要被序列平稳化消除的非平稳信息 $\sigma_{\mathbf{x}}, \mu_{\mathbf{Q}}, \mathbf{K}$ 。

**去平稳注意力** 为了恢复原始非平稳序列上的注意力，我们尝试将消失的非平稳性信息带回其计算中。基于公式(5) ，关键在于近似正缩放标量 $\tau = \sigma_{\mathbf{x}}^2 \in \mathbb{R}^{+}$ 和平移向量 $\boldsymbol{\Delta} = \mathbf{K}\mu_{\mathbf{Q}} \in \mathbb{R}^{S \times 1}$ ，我们将其定义为去平稳因子。由于深度模型严格的线性性质很难成立，除了估计和利用真实因子外，我们尝试通过一个简单而有效的多层感知器，直接从非平稳序列 $\mathbf{x}$ 的统计量 $\mathbf{Q}$ 和 $\mathbf{K}$ 中学习去平稳因子。由于从 $\mathbf{Q}', \mathbf{K}'$ 中只能发现有限的非平稳性信息，补偿非平稳性的唯一合理来源是未被归一化的原始 $\mathbf{x}$ 。因此，为了直接从公式(5) 中学习去平稳因子，我们应用多层感知器独立学习未平稳化 $\mathbf{x}$ 的统计量 $\mu_{\mathbf{x}}, \sigma_{\mathbf{x}}$ 来估计 $\tau, \boldsymbol{\Delta}$ 。去平稳注意力计算如下：
$$
\begin{align*}
\log \tau &= \mathrm{MLP}(\sigma_{\mathbf{x}}, \mathbf{x}), \quad \boldsymbol{\Delta} = \mathrm{MLP}(\mu_{\mathbf{x}}, \mathbf{x}) \\
\mathrm{Attn}(\mathbf{Q}', \mathbf{K}', \mathbf{V}', \tau, \boldsymbol{\Delta}) &= \mathrm{Softmax}\left( \frac{\tau \mathbf{Q}'\mathbf{K}'^{\top} + \mathbf{1}\boldsymbol{\Delta}^{\top}}{\sqrt{d_k}} \right) \mathbf{V}'
\tag{6}
\end{align*}
$$
其中去平稳因子 $\tau$ 和 $\boldsymbol{\Delta}$ 由所有层的去平稳注意力共享（见图2）。去平稳注意力机制从平稳化序列 $\mathbf{Q}', \mathbf{K}'$ 和非平稳序列 $\mathbf{x}, \mu_{\mathbf{x}}, \sigma_{\mathbf{x}}$ 中学习时间依赖性，并与平稳化后的值 $\mathbf{V}'$ 相乘。因此，它可以同时利用平稳化序列的可预测性和保持原始序列的固有时间依赖性。

#### 2.1.3 整体架构

![20250423222116](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250423222116.png)

遵循先前在时间序列预测中使用 Transformer [39, 37] 的做法，我们采用标准的编码器 - 解码器结构（图2），其中编码器用于从过去的观测中提取信息，解码器用于聚合过去的信息并从简单的初始化中优化预测。标准的非平稳 Transformer 是在普通 Transformer [34] 的输入和输出上应用序列平稳化，并将自注意力替换为我们提出的去平稳注意力，这可以提升基础模型对非平稳序列的预测能力。对于 Transformer 变体 [19, 39, 37] ，我们用去平稳因子 $\tau, \boldsymbol{\Delta}$ 变换 Softmax 内部的项，以重新整合非平稳信息（实现细节见附录E.2 ）。

## 三、实验验证与结果分析

### 3.1 消融实验

![20250423232525](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250423232525.png)

**质量评估** 为探究我们提出的框架中每个模块的作用，我们比较了三个模型在ETTm2数据集上的预测结果：普通Transformer、仅含序列平稳化的Transformer，以及我们的非平稳Transformer。从图3中我们发现，这两个模块从不同角度增强了Transformer对非平稳数据的预测能力。序列平稳化专注于对齐每个序列输入之间的统计属性，这极大地帮助Transformer在分布外的数据上进行泛化。然而，如图3(b)所示，训练时过度平稳化的情况使得深度模型更倾向于输出具有显著高平稳性的无变化序列，而忽略了实际非平稳数据的特性。借助去平稳注意力，模型重新关注到实际时间序列的固有非平稳性。这有助于准确预测序列的详细变化，这在实际时间序列预测中至关重要。

![20250423232612](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250423232612.png)

**定量性能** 除了上述案例研究，我们还提供了与平稳化方法的定量预测性能比较：深度方法RevIN [17] 和序列平稳化（3.1节）。如表5所示，RevIN和序列平稳化辅助下的预测结果基本相同，这表明我们框架中无参数版本的归一化在使时间序列平稳化方面表现良好。此外，非平稳Transformer中提出的去平稳注意力进一步提升了性能，在六个基准数据集上都取得了最佳效果。去平稳注意力带来的均方误差（MSE）降低显著，尤其是在数据集具有高度非平稳性时（Exchange数据集：从0.569降至0.461；ETTm2数据集：从0.461降至0.306 ）。比较结果表明，简单地对时间序列进行平稳化仍然限制了Transformer的预测能力，而非平稳Transformer中的互补机制能够恰当地释放模型对非平稳序列的预测潜力。

### 3.2 模型分析

![20250423232728](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250423232728.png)
>图4：相对平稳性计算为模型预测值与真实值的ADF检验统计量之比。从左到右，数据集的非平稳性逐渐增强。仅采用平稳化的模型倾向于输出高度平稳的序列，而我们的方法给出的预测结果的平稳性更接近真实值。

**过度平稳化问题** 为从统计角度验证过度平稳化问题，我们分别用上述方法训练Transformer，将所有预测的时间序列按时间顺序排列，并与真实值比较平稳度（图4）。仅采用平稳化方法的模型倾向于输出具有意外高平稳度的序列，而去平稳注意力辅助下的结果更接近实际值（相对平稳度在[97%, 103%] 范围内）。此外，随着序列平稳度的增加，过度平稳化问题变得更加明显。平稳度的巨大差异可以解释仅采用平稳化的Transformer性能较差的原因。这也表明，作为一种内部改进，去平稳注意力缓解了过度平稳化问题。

![20250423232916](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250423232916.png)
>表6：框架设计的消融实验。Baseline指普通Transformer，Stationary指添加序列平稳化，DeFF指在前馈层重新整合非平稳性，DeAttn指通过去平稳注意力重新整合，Stat + DeFF指添加序列平稳化并在前馈层重新整合，Stat + DeAttn指我们提出的框架。

**非平稳信息重新整合探究** 值得注意的是，通过将过度平稳化定义为难以区分的注意力，我们将设计空间聚焦于注意力计算机制。为探索检索非平稳信息的其他方法，我们进行了将均值 $\mu$ 和标准差 $\sigma$ 重新整合到前馈层（DeFF）的实验，DeFF位于Transformer架构的左侧部分。具体来说，我们将学习到的 $\mu$ 和 $\sigma$ 迭代输入到每个前馈层。如表6所示，仅在输入被平稳化（Stationary）时，重新整合非平稳性才是必要的，这对预测有益，但会导致模型输出的平稳性差异。我们提出的设计（Stat + DeAttn）进一步提升了性能，在大多数情况下（77%）取得了最佳效果。除了理论分析，实验结果进一步验证了我们在注意力中重新整合非平稳性设计的有效性。
