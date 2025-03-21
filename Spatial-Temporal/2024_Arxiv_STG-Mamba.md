# STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model

>领域：时空序列预测  
>发表在：Arxiv  
>模型名字：**S**patial-**T**emporal **G**raph Mamba  
>文章链接：[STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model](https://arxiv.org/abs/2403.12418)
>代码仓库：[[STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model](https://github.com/LincanLi98/STG-Mamba)](<https://github.com/LincanLi98/STG-Mamba>)
![20250321103146](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321103146.png)

## 一、研究背景与问题提出

### 1.1 研究现状

时空图数据是一类非欧几里得数据，广泛存在于日常生活中，如城市交通网络、地铁系统的流入/流出量、社交网络、区域能源负荷、气象观测等。由于时空图数据具有动态、异质和非平稳的特性，准确且高效的时空图预测长期以来一直是一项具有挑战性的任务。  

近来，随着Mamba的普及[Gu and Dao, 2023; Wang et al., 2024; Liu et al., 2024]，现代选择性状态空间模型（SSSM）在计算机视觉和自然语言处理领域的研究者中引起了广泛关注。SSSM是状态空间模型的一个变体，其起源于控制科学与工程领域[Lee et al., 1994; Friedland, 2012]。状态空间模型通过一组由一阶微分或差分方程关联的输入、输出和状态变量，提供了描述物理系统动态状态演化的专业框架，允许以紧凑的方式对多输入多输出（MIMO）系统进行建模与分析[Aoki, 2013]。  

时空图学习可被视为理解和预测时空图网络演化的复杂过程，与状态空间转换过程高度相似。基于深度学习的SSSM为时空图学习带来新视野，但在适配其架构进行时空图建模时仍面临巨大挑战。受SSSM长程建模能力和低计算开销的启发，本文提出时空学习Mamba（STG-Mamba），主要贡献包括：  

- **首次适配SSSM到时空图学习**：按堆叠残差编码器模式扩展SSSM，以图选择性状态空间块（GS3B）为模块，结合卡尔曼滤波图神经网络（KFGN）、时空选择性状态空间模块（ST-S3M）及多流前馈连接，协调不同模块。  
- **提出ST-S3M模块**：实现输入依赖的自适应时空图特征选择，利用图选择性扫描算法接收KFGN图信息，辅助更新状态转移矩阵和控制矩阵。  
- **引入KFGN方法**：作为自适应时空图生成与更新模块，通过DynamicFilter-GNN生成动态图结构，KF-Upgrading机制整合不同时间粒度输入，借助卡尔曼滤波统计理论处理输出嵌入。  
- **广泛评估验证性能**：在三个开源时空图数据集上验证，STG-Mamba不仅预测性能超越基准方法，还实现O(n)计算复杂度，显著降低计算开销。

## 二、问题剖析与解决策略

### 2.0 Preliminaries

#### 时空图系统（STG系统）  

我们首次将由时空图数据构成的网络定义为时空图系统。在理论层面，系统是物理过程的表征，包含描述系统当前状态的状态变量、影响系统状态的输入变量，以及反映系统响应的输出变量。对于时空图系统，该框架经适配以涵盖空间依赖和时间演化特性，通过代表空间实体的节点、表示空间交互的边，以及随时间演化的状态变量构建系统，从而捕捉时空图数据的动态特征。  

#### 状态空间模型（SSMs）  

基于深度学习的 SSMs 是一类新提出的序列模型，与 RNN 架构和经典状态空间模型联系紧密。其核心是通过特定连续系统模型，将多维输入序列通过隐式潜在状态表示映射为对应输出序列。SSMs 由四个参数 $(\mathbf{A, B, C, D})$ 定义，这些参数决定输入（控制信号）和当前状态如何确定下一状态与输出。该框架通过支持线性和非线性计算，实现高效序列建模。作为 SSMs 的变体，SSSMs 着重于在 SSMs 上构建选择机制，这与注意力机制的核心思想高度相似，使其成为 Transformer 架构的有力竞争者。  

#### 基于 SSSM 的时空图预测  

将 SSSMs 用于时空图预测时，问题可描述为：动态识别并利用历史时空数据与图结构的相关部分，以预测时空图系统的未来状态。  
给定时空图 $\mathbb{G}^{ST} = (V^{ST}, E^{ST}, A^{ST})$ 和历史时空序列数据 $\mathrm{X_{t-p+1:t}} = \{ \mathrm{X_{t-p+1}, X_{t-p+2}, ..., X_t} \}$，目标是利用 SSSMs 预测未来时空图系统的状态 $\hat{\mathrm{X}}_{t+1:t+k} = \{ \hat{\mathrm{X}}_{t+1}, ..., \hat{\mathrm{X}}_{t+k} \}$。这一过程通过学习映射函数 $F_{SSSM}(\cdot)$ 实现，该函数动态选择相关的状态转移和交互用于预测：  
$$F_{SSSM}(\mathrm{X_{t-p+1:t}}; \mathbb{G}^{ST}) = \hat{\mathrm{X}}_{t+1:t+k} \tag{1}$$

### 2.1 解决方法

#### 2.1.1 卡尔曼滤波图神经网络

在构建图神经网络（GNN）时采用基于卡尔曼滤波方法的动机，源于增强时空预测可靠性与准确性的需求。时空图（STG）大数据（如交通传感器记录、气象站记录等）通常包含固有偏差与噪声，而现有方法往往忽略这些问题。通过整合基于卡尔曼滤波的优化与升级，卡尔曼滤波图神经网络（KFGN）通过动态权衡不同时间粒度数据流的可靠性，基于估计方差优化这些数据流的融合，有效解决了这些不准确性。该方法不仅纠正了数据集的固有误差，还显著提升了模型捕捉 STG 模式中复杂依赖关系的能力。  

如图 2 所示，KFGN 流程包含两个关键步骤。第一步，将不同时间粒度（如近期步长/周期步长/趋势步长）的模型输入生成的嵌入，送入 DynamicFilter-GNN 模块处理。第二步，通过卡尔曼滤波升级模块整合并优化 DynamicFilter-GNN 模块的输出。在 DynamicFilter-GNN 的初始阶段，定义一个大小为 $\mathbb{R}^{\text{in\_fea} \times \text{in\_fea}}$ 的可学习基滤波器参数矩阵，用于变换图邻接矩阵，从而动态调整节点间的连接程度。随后，统一初始化权重和偏置，初始化标准差 $\text{stdv}$ 计算为输入特征数量平方根的倒数：$\text{stdv} = \frac{1}{\sqrt{\text{in\_fea}}}$。这种统一初始化对确保模块从“中性起点”开始至关重要。  

![20250321111045](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321111045.png)

随后，通过线性变换层对基滤波器进行变换，并与原始邻接矩阵 $A_{\text{ini}}^{c_i}$ 结合，得到动态调整的邻接矩阵 $A_{\text{dyn}}^{c_i}$。令 $c_{i, i=\{1,2,3\}} \in \{r, p, q\}$ 表示不同时间粒度的输入数据（如 $c_1 = r$ 表示近期历史数据，$c_2 = p$ 表示周期历史数据，$c_3 = q$ 表示趋势历史数据）。利用 $A_{\text{dyn}}^{c_i}$、输入嵌入 $h_{\text{in}}^{c_i}$、权重 $W_{DF}^{c_i}$ 和偏置 $b_{DF}^{c_i}$，输入嵌入进行图卷积：  
$$  
h_{DF}^{c_i} = h_{\text{in}}^{c_i} \cdot (A_{\text{dyn}}^{c_i} \cdot W_{DF}^{c_i}) + b_{DF}^{c_i} \quad (2)  
$$  
该设计使模型能基于学习到的调整，动态调整图中节点间的连接强度。  

在 DynamicFilter-GNN 之后，下一步是卡尔曼滤波升级。由于输入数据来自同一数据集但时间粒度不同，假设其服从高斯分布：  
$$  
\begin{aligned}  
y_q(x; \mu_q, \sigma_q) &\triangleq \frac{\exp^{-\frac{(x - \mu_q)^2}{2\sigma_q^2}}}{\sqrt{2\pi\sigma_q^2}}; &y_p(x; \mu_p, \sigma_p) &\triangleq \frac{\exp^{-\frac{(x - \mu_p)^2}{2\sigma_p^2}}}{\sqrt{2\pi\sigma_p^2}}; &y_r(x; \mu_r, \sigma_r) &\triangleq \frac{\exp^{-\frac{(x - \mu_r)^2}{2\sigma_r^2}}}{\sqrt{2\pi\sigma_r^2}}; \quad (3)  
\end{aligned}  
$$  
其中下标 $q = c_3, p = c_2, r = c_1$ 分别表示历史趋势/周期/近期数据集。  

采用卡尔曼滤波方法从多粒度观测集中推导准确信息。具体而言，通过乘法整合各分支的概率分布函数：  
$$  
\begin{aligned}  
y_{fuse}(x; \mu_q, \sigma_q, \mu_p, \sigma_p, \mu_r, \sigma_r) &= \frac{\exp^{-\frac{(x - \mu_q)^2}{2\sigma_q^2}}}{\sqrt{2\pi\sigma_q^2}} \times \frac{\exp^{-\frac{(x - \mu_p)^2}{2\sigma_p^2}}}{\sqrt{2\pi\sigma_p^2}} \times \frac{\exp^{-\frac{(x - \mu_r)^2}{2\sigma_r^2}}}{\sqrt{2\pi\sigma_r^2}} \\  
&= \frac{1}{(2\pi)^{3/2}\sqrt{\sigma_q^2\sigma_p^2\sigma_r^2}} \exp^{-\left( \frac{(x - \mu_q)^2}{2\sigma_q^2} + \frac{(x - \mu_p)^2}{2\sigma_p^2} + \frac{(x - \mu_r)^2}{2\sigma_r^2} \right)} \quad (4)  
\end{aligned}  
$$  
通过重组公式（4）为简化版本，得到：  
$$  
y_{fuse}(x; \mu_{fuse}, \sigma_{fuse}) = \frac{1}{\sqrt{2\pi\sigma_{fuse}^2}} \exp^{-\frac{(x - \mu_{fuse})^2}{2\sigma_{fuse}^2}} \quad (5)  
$$  
其中 $\mu_{fuse} = \frac{\mu_q / \sigma_q^2 + \mu_p / \sigma_p^2 + \mu_r / \sigma_r^2}{1 / \sigma_q^2 + 1 / \sigma_p^2 + 1 / \sigma_r^2}$，$\sigma_{fuse}^2 = \frac{1}{1 / \sigma_q^2 + 1 / \sigma_p^2 + 1 / \sigma_r^2}$。  

为简化 $\mu_{fuse}$ 和 $\sigma_{fuse}^2$ 的表达，引入参数 $\omega_q = \frac{1}{\sigma_q^2}, \omega_p = \frac{1}{\sigma_p^2}, \omega_r = \frac{1}{\sigma_r^2}$，则 $\mu_{fuse}$ 和 $\sigma_{fuse}^2$ 可重写为：  
$$  
\mu_{fuse} = \frac{\mu_q \omega_q + \mu_p \omega_p + \mu_r \omega_r}{\omega_q + \omega_p + \omega_r} ; \quad \sigma_{fuse}^2 = \frac{1}{\omega_q + \omega_p + \omega_r} \quad (6)  
$$  
这意味着不同分支的观测可通过加权和有效整合，权重是方差的组合：  
$$  
\begin{aligned}  
\mu_{fuse} &= \mu_q \left( \frac{\omega_q}{\omega_q + \omega_p + \omega_r} \right) + \mu_p \left( \frac{\omega_p}{\omega_q + \omega_p + \omega_r} \right) + \mu_r \left( \frac{\omega_r}{\omega_q + \omega_p + \omega_r} \right) \\  
\Downarrow \\  
y_{fuse} &= y_q \left( \frac{\omega_q}{\omega_q + \omega_p + \omega_r} \right) + y_p \left( \frac{\omega_p}{\omega_q + \omega_p + \omega_r} \right) + y_r \left( \frac{\omega_r}{\omega_q + \omega_p + \omega_r} \right) \quad (7)  
\end{aligned}  
$$  
直接计算观测集的方差需要完整数据，计算成本高。因此，通过计算每个训练样本的方差分布来估计方差：  
$$  
\mathbb{E}[\sigma_{\{q,p,r\}}^2] = \frac{1}{N_m} \sum_i \frac{(S_i - \bar{S})^2}{L} \quad (8)  
$$  
其中 $L$ 是每个样本序列的长度，$N_m$ 是数据样本数量，$S_i$ 表示第 $i$ 个观测值，$\bar{S}$ 表示所有观测样本的平均值。为进一步改进整合与升级，添加两个可学习权重参数 $\epsilon, \varphi$ 平衡不同观测分支。基于公式（7）中 $y_{fuse}$ 的表达式，卡尔曼滤波升级模块的输出为：  
$$  
\tilde{y}_{fuse} = \frac{\epsilon \cdot (\hat{y}_q \omega_q) + \varphi \cdot (\hat{y}_p \omega_p) + \hat{y}_r \omega_r}{\omega_q + \omega_p + \omega_r} \quad (9)  
$$  
其中 $\omega_q = \frac{1}{\sigma_q^2}, \omega_p = \frac{1}{\sigma_p^2}, \omega_r = \frac{1}{\sigma_r^2}$，$\epsilon$ 和 $\varphi$ 是可学习权重。最后，为便于神经网络训练并确保可扩展性，移除公式（9）中的常数分母 $(\omega_q + \omega_p + \omega_r)$，卡尔曼滤波升级的最终输出定义为：  
$$  
\hat{y}_{fuse} = \epsilon \cdot (\hat{y}_q \omega_q) + \varphi \cdot (\hat{y}_p \omega_p) + \hat{y}_r \omega_r \quad (10)  
$$  
此处，$\hat{y}_{fuse}$ 是 KF 升级模块的最终输出。请注意，我们的方法是经典卡尔曼滤波的优化版本，专门为基于深度学习的方法设计，确保计算效率、动态分层 STG 特征融合和准确性。

#### 2.1.2 ST-S3M：时空选择性状态空间模块

![20250321130632](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321130632.png)

ST-S3M 在自适应时空图（STG）特征选择中发挥重要作用，相当于注意力机制。其架构如图 1 所示。经 KFGN 模块处理动态时空依赖建模和基于统计的集成与升级后，生成的嵌入 $\hat{h}_{fuse}$ 被送入 ST-S3M，进行输入特定的动态特征选择。在 ST-S3M 中，输入的 STG 嵌入先经过线性层，再执行拆分操作。设 $b$ 为批量大小，$l$ 为序列长度，$d_{model}$ 为特征维度，$d_{inner}$ 为模型内部特征维度，公式如下：  
$$  
\begin{align}  
h_{\text{main-res}} &= W_{in} \hat{h}_{fuse} + b_{in} \\  
(h_{\text{main}}, res) &= \text{split}(h_{\text{main-res}}) \quad (11)  
\end{align}  
$$  
其中 $\hat{h}_{fuse} \in \mathbb{R}^{b \times l \times d_{model}}$，$W_{in} \in \mathbb{R}^{2d_{inner} \times d_{model}}$，$b_{in} \in \mathbb{R}^{2d_{inner}}$，$h_{\text{main-res}} \in \mathbb{R}^{b \times l \times 2d_{inner}}$，$h_{\text{main}}, res \in \mathbb{R}^{b \times l \times d_{inner}}$。  

接着，$h_{\text{main}}$ 流入一维卷积层，再经过 SiLU 激活函数：  
$$  
h'_{\text{main}} = \text{SiLU}(\text{Conv1D}(h_{\text{main}})) \quad (12)  
$$  
其中 $h'_{\text{main}} \in \mathbb{R}^{b \times l \times d_{inner}}$。SiLU 输出被送入图状态空间选择机制：  
$$  
h_{\text{sssm}} = \text{GSSSM}(h'_{\text{main}}) \quad (13)  
$$  
其中 $h_{\text{sssm}} \in \mathbb{R}^{b \times l \times d_{inner}}$。同时，残差部分 $res$ 也经过 SiLU 处理，最后通过元素级融合以对抗主 STG 嵌入及其激活：$h_{\text{sssm}} \odot \text{SiLU}(res)$。融合结果通过线性投影变换：  
$$  
h_{out} = W_{out}(h_{\text{sssm}} \odot \text{SiLU}(res)) + b_{out} \quad (14)  
$$  
其中 $h_{out} \in \mathbb{R}^{b \times l \times d_{model}}$ 是 ST-S3M 的最终输出，$W_{out} \in \mathbb{R}^{d_{model} \times d_{inner}}$，$b_{out} \in \mathbb{R}^{d_{model}}$。  

ST-S3M 中的 GSSSM 在自适应时空特征选择中起主要作用。我们在算法 2 中详细说明图状态空间选择机制的参数计算和更新过程。图选择扫描算法（算法 1）在前向传播中接收图信息，是状态空间选择过程中最重要的步骤，详述如下。  

##### 图选择扫描算法  

该算法是基本选择扫描的扩展，将 KFGN 生成的动态图信息集成到状态空间选择与更新过程，增强 Mamba 捕捉 STG 依赖的能力。关键步骤和修改如下：  

1. 首先获取输入张量 $u \in \mathbb{R}^{(b, l, d_{in})}$ 的维度，其中 $b$ 为批量大小，$l$ 为序列长度，$d_{in}$ 为输入特征维度；$\mathbf{A}$ 的第二维记为 $n$。  
2. 突出图选择扫描算法的核心创新——集成的图信息前馈（对应算法 1 第 2–5 行）。从 DynamicFilter-GNN 检索动态图邻接矩阵 $\alpha_t$：$\alpha_t = \text{DynamicFilter-GNN.get\_transformed\_adjacency()}$。为使图信息参与状态空间选择，融合参数 $\Delta^*$，但 $\alpha_t$ 与 $\Delta^*$ 维度可能不一致，因此初始化填充矩阵 $\text{adj\_padded} = \mathbf{1}^{d_{in} \times d_{in}}$，用 $\alpha_t$ 的图信息填充：$\text{adj\_padded}[:\alpha_t.size(0), :\alpha_t.size(1)] = \alpha_t$。维度调整后，通过矩阵乘法集成 $\Delta^*$ 和 $\text{adj\_padded}$：$\Delta' = \text{matmul}(\Delta^*, \text{adj\_padded})$。  
3. 离散化连续参数 $\mathbf{A}$ 和 $\mathbf{B}$：  

$$  
\begin{align}  
\text{状态转移矩阵更新: } &\text{deltaA} = \exp(\text{einsum}(\Delta', \mathbf{A})) \quad (15) \\  
\text{控制矩阵更新: } &\text{deltaB}_u = \text{einsum}(\Delta', \mathbf{B}, u)  
\end{align}  
$$  
其中 einsum 表示爱因斯坦求和约定，$\text{deltaA}$ 是更新后的状态转移矩阵，$\text{deltaB}_u$ 是更新后的控制矩阵。  
4. 对状态 $x \in \mathbb{R}^{b \times d_{in} \times n}$ 执行迭代状态更新。对时间步 $i$ 到 $l$，有：  
$$  
\begin{align}  
x &\leftarrow \text{deltaA}[:, i] \times x + \text{deltaB}_u[:, i] \quad (16) \\  
z &\leftarrow \text{einsum}(x, \mathbf{C}[:, i, :])  
\end{align}  
$$  
其中 $z$ 是通过爱因斯坦求和约定计算的当前输出，添加到输出列表 $z_s$。堆叠 $z_s$ 形成输出张量 $z$，最后将直接增益 $\mathbf{D}$ 加入最终输出：$z \leftarrow z + u \times \mathbf{D}$。  

图选择扫描算法具有增强时空依赖建模、适应变化图结构、提高预测精度等优势，特别适用于时空图学习任务。

### 2.2 模型结构

![20250321103146](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321103146.png)
> 图1展示了所提出的STG-Mamba的完整架构。具体而言，我们按残差编码器模式构建整体架构，以实现高效序列数据建模与预测。在STG-Mamba中，我们将图选择性状态空间块 Graph Selective State Space Block（GS3B）作为基础编码器模块，重复N次。GS3B由多个网络与操作组成，包括层归一化、卡尔曼滤波图神经网络（KFGN，由DynamicFilter-GNN及后续的卡尔曼滤波更新（KF-Upgrading）构成）、时空选择性状态空间模块（ST-S3M，包含Linear & Split、Conv1D、SiLU、图状态空间选择机制（GSSSM）、逐元素拼接、图信息前馈），以及残差连接。此处的图信息前馈结构专门用于协调不同模块的信息传输与更新，确保每个模块都能获取最新的时空图信息。

## 三、实验验证与结果分析

### 3.1 性能实验

![20250321135813](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321135813.png)

### 3.2 鲁棒性实验

时空图（STG）数据具有显著的周期性和多样性，城市交通流动性/交通数据显示出早晚高峰出行时间与非高峰时段之间，以及工作日和周末之间的明显差异。鉴于这些因外部环境变化而产生的差异，确定深度学习模型是否能够在不同条件下有效地对时空依赖性进行建模具有重要意义。因此，我们建立了四种不同的外部场景：(a) 工作日上午8:00 - 11:00和下午4:00 - 7:00的高峰时段；(b) 工作日的非高峰时段；(c) 周末（全天）；以及(d) 非周末（全天）。我们在PeMS04数据集上进行了大量实验。表3展示了这四种交通场景的预测结果。

![20250321140140](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321140140.png)

一个理想的基于深度学习的时空图系统应该对外部干扰具有鲁棒性，在不同场景下保持一致的性能。与ASTGNN、PDFormer和STAEformer等方法相比，高峰时段和非高峰时段场景之间的性能差异显著。即使在周末/非周末场景中，STG-Mamba也表现出最小的性能差异，显示出最大的鲁棒性，并在RMSE（均方根误差）/MAE（平均绝对误差）/MAPE（平均绝对百分比误差）指标上取得了最佳成绩。总之，STG-Mamba表现出卓越的鲁棒性，在各种交通条件下的性能优于现有的基线模型。

在时空预测中，基于统计的评估是必不可少的，因为它们为模型在时间和空间上捕捉和预测复杂数据动态的能力提供了一种可量化的衡量标准。具体而言，$R^2$ 和 $\Delta_{\text{VAR}}$ 有助于评估模型在不同条件下的准确性和鲁棒性，确保预测的可靠性，并有效地为决策过程提供信息。统计评估结果如表4所示。在这里，STG-Mamba在所有数据集上都表现出卓越的性能，取得了最高的 $R^2$ 值和最低的 $\Delta_{\text{VAR}}$ 分数，表明在处理时空依赖性方面准确性和效率的提升。

![20250321140306](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321140306.png)

### 消融实验

![20250321140338](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321140338.png)

为了研究STG-Mamba中每个模型组件的有效性，我们进一步设计了五种模型变体，并在PeMS04/杭州地铁（HZMetro）/KnowAir数据集上评估它们的预测性能：(I) STG-Mamba：未做任何修改的完整STG-Mamba模型。(II) 无卡尔曼滤波升级（KF-Upgrading）的STG-Mamba：我们用简单的求和平均操作替换了KFGN中的KF-Upgrading模块，以融合三个时间分支。(III) 无动态滤波器线性层（Dynamic Filter Linear）的STG-Mamba：我们用基本的静态图卷积替换了KFGN中提出的Dynamic Filter Linear。(IV) 无图状态空间选择机制（Graph State Space Selection Mechanism，GSSSM）的STG-Mamba：GSSSM被基本的状态空间选择机制（SSSM）[Gu和Dao，2023]所取代。(V) 无时空自监督状态空间模块（ST-S3M）的STG-Mamba：整个ST-S3M模块从GS3B编码器中被移除。

如图3所示，无KF-Upgrading的STG-Mamba表现比完整模型差，这表明了对于基于SSSM的方法，采用基于统计学习的图神经网络（GNN）状态空间升级和优化的必要性和适用性。由于缺少Dynamic Filter Linear而导致的性能下降证明了它的有效性，以及为STG特征学习设计合适的自适应GNN的必要性。此外，在ST-S3M中移除状态空间选择机制会导致模型能力的大幅下降，证明了使用SSSM作为注意力机制替代方案的可行性。最后，移除ST-S3M会使STG-Mamba退化为一个普通的基于GNN的模型，从而导致性能最低。
