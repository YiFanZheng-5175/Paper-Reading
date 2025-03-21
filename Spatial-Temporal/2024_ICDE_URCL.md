# A Unified Replay-based Continuous LearningFramework for Spatio-Temporal Predictionon Streaming Data

>领域：时空预测，持续学习  
>发表在：2025 ICDE  
>模型名字：**U**nified **R**eplay-based **C**ontinuous **L**earnin  
>文章链接：[A Unified Replay-based Continuous LearningFramework for Spatio-Temporal Predictionon Streaming Data](https://arxiv.org/abs/2404.14999)  
>代码仓库： None  
![20250321141716](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321141716.png)

## 一、研究背景与问题提出

### 1.1 研究现状

社会进程的持续数字化以及传感技术的相应部署产生了越来越多的时空数据。例如，道路传感器群体提供的数据可以捕捉不同地点随时间变化的交通流量。此外，应用程序越来越多地接收持续生成的大量时空数据，这被称为流式时空数据。来自道路传感器的数据就是这类数据的一个例子。

在这项研究中，我们关注对流式时空数据进行预测的新问题，总体目标是从具有时空相关性的流式时空数据中学习一个模型，同时保留已学习的历史知识并捕捉时空模式，以准确预测未来的时空观测值。

许多时空预测应用，如交通流量预测[1]-[3]、交通速度预测[4]、[5]和按需服务预测[6]，都伴随着经过精心定制的预测模型。这些模型使用各种方法来预测时空数据，包括基于传统统计学的方法[7]、[8]以及基于卷积[9]-[11]和循环[4]、[12]神经网络的方法。然而，现有的模型是静态训练的，无法处理流式数据。静态模型通常只训练一次以拟合特定的数据集，然后在两个数据集遵循相同分布的潜在假设下，对另一个数据集进行预测。然而，概念漂移（即数据分布随时间变化）在流式时空数据中经常发生。因此，将静态模型直接应用于流式数据可能会导致显著的性能下降[13]。

因此，我们需要一种新的持续学习（CL）模型，它能够不断适应所看到的数据，并且能够随着时间的推移持续学习时空预测任务。然而，开发这种模型并非易事，原因如下几个挑战。

- 挑战一：灾难性遗忘。在时空预测的持续学习中，缓解灾难性遗忘是一项挑战。灾难性遗忘是指当静态模型简单地使用新到达的数据重新训练时，会突然忘记先前学习的知识的倾向[13]。当模型基于传入的数据（由于概念漂移，其分布不同）持续重新训练时，对先前任务的预测性能会下降[14]。这是因为模型总是基于在一个时期内获得的新数据重新训练，然后用于预测另一个时期的数据。当数据持续到达时，模型学习到的知识也在不断变化。尽管已经做出了很多努力，使用持续学习技术来解决计算机视觉[15]、[16]和自然语言处理[17]中的灾难性遗忘问题，但由于时空数据的独特特征，这些技术无法直接应用于时空预测。具体来说，目前还不存在能够有效捕捉流式时空数据中的空间依赖性或时间相关性的持续学习模型。

- 挑战二：时空数据和预测应用的多样性。发现不同时空数据和预测应用的共性仍然是一个关键问题。尽管存在各种经过精心定制的静态模型，但将每个这样的模型转换为其持续版本既耗时又不现实，因为每个时空预测模型都有不同的具体设置（例如，网络架构、数据集和目标函数）需要考虑。因此，在许多流式时空预测应用中实现一个准确且高效的模型是非常值得期待的，但也并非易事。

- 挑战三：整体特征保留。对于现有模型来说，在流式数据上学习用于时空预测的整体特征是一项挑战。具体来说，整体特征保留了多个时间段之间的语义相似性。在持续的时空预测中，保留先前学习的语义特征可能有助于未来的预测。例如，在之前工作日的非高峰时段学习到的交通模式可能有助于对后续工作日的预测。大多数现有的持续学习模型专注于为当前任务学习判别性特征，而忽略了先前学习的可能对未来任务有用的特征[18]，这通常会导致未来预测的结果不理想，从而对持续的时空预测产生不利影响。

## 二、问题剖析与解决策略

本文通过提供一个基于统一重放的持续学习 Unified Replay-based Continuous Learnin（URCL）框架来应对上述挑战，用于对流式数据进行时空预测。URCL框架包括三个主要模块：数据集成、时空持续表示学习 spatio-
temporal continuous representation learni（STCRL）和时空预测。

- 为了缓解灾难性遗忘（挑战一），我们提出了一种时空混合（STMixup）机制，将当前采样的时空观测值与从重放缓冲区中选择的样本进行融合，重放缓冲区存储了先前学习的观测值的一个子集。我们还提出了一种基于排名的最大干扰检索采样策略，以便从重放缓冲区中选择有代表性的样本。

- 为了支持多样化的时空数据和预测应用（挑战二），我们发现了现有通常基于自动编码器架构的方法的共性。我们提出了一种新颖的时空预测网络，包括一个时空编码器（STEncoder）和一个时空解码器（STDecoder），以捕捉复杂的时空相关性，从而实现准确性。

- 为了解决整体特征保留的问题（挑战三），我们提出了一个时空简单孪生（STSimSiam）网络，以避免在STCRL模块中出现整体特征丢失。具体来说，STSimSiam网络包含两个STEncoder和一个投影多层感知器（MLP）头，其中STEncoder与时空预测网络中的STEncoder是共享的。我们首先使用互信息最大化来确保在流式时空预测中保留整体特征。此外，考虑到数据增强可以帮助模型学习更有效的表示[19]、[20]，我们在探索时空数据的独特特征的基础上，提供了五种时空数据增强方法，从而实现有效的整体时空特征学习。

### 2.0 问题定义

硬件和无线网络技术的进步催生了多功能传感器设备[46]。这一发展使得由散布在大片地理区域的微小传感器节点组成的系统（称为传感器网络）能够记录流式时空数据。

- 定义1（传感器网络）：传感器网络用图$G = (V, E)$表示，其中$V$是传感器节点集，$E$是边集。每个节点$v_i \in V$代表一个传感器，每条边$e_{i,j} \in E$表示传感器$v_i$和$v_j$之间的连接性。

- 定义2（时空观测）：给定一个传感器网络$G$，所有传感器在时间槽$t$（例如，上午9:00 - 上午9:15）以采样间隔$\Delta t$（例如，15分钟）收集的时空观测值用$X_t \in \mathbb{R}^{|V|\times C}$表示，其中$|V|$表示传感器的数量，$C$是节点特征的维度（例如，交通流量和速度）。

在本文的其余部分，当从上下文中可以清楚其含义时，我们将使用“时空观测”的简称“观测”。

- 定义3（流式时空数据序列）：给定一个传感器网络$G$和一个包含$n$个连续时间槽的时间段$\mathbb{T}_i$，即$\mathbb{T}_i = \langle t_i^1, t_i^2, \cdots, t_i^n\rangle$，一个流式时空数据序列是一个矩阵序列，每个矩阵表示在具有一个采样间隔的特定时间槽$t_i^j$的观测值，其中$t_i^{j + 1} - t_i^j = \Delta t (1 \leq j \leq n - 1)$。特别地，一个流式时空数据序列$D_i$是一个观测序列$D_i = \langle X_{t_i^1}, X_{t_i^2}, \cdots, X_{t_i^n}\rangle$，其中$n$也是序列长度。

基于上述定义，我们正式定义流式时空预测（SSTP）问题如下。

SSTP问题：考虑一个传感器网络$G$，它发出一个流式时空数据序列$\mathbb{D} = \langle D_1, D_2, \cdots, D_m\rangle (m \geq 1)$，其中$D_i$表示在时间段$\mathbb{T}_i$内的观测序列，且$|D_i| = n$（例如，$\mathbb{T}_i$是一天，当间隔为15分钟时$n$为96）。给定包含在$D_i (1 \leq i \leq m)$中的当前观测值$X_{t_i^j}$，SSTP问题旨在学习一个函数$f_i(\cdot)$，以便基于当前观测值及其之前的$M - 1 (M < n)$个历史观测值来预测$N$个未来观测值，同时最大程度地保留从之前的流式数据序列$\langle D_1, \cdots, D_{i - 1}\rangle$中学习到的知识，即：
$$
\overbrace{[\cdots, X_{t_i^{j - 1}}, X_{t_i^j}; G]}^{M\text{ 个观测值}} \overset{f_i(\cdot)}{\longrightarrow} \overbrace{[X_{t_i^{j + 1}}, X_{t_i^{j + 2}}, \cdots]}^{N\text{ 个观测值}}
$$
其中，从$\langle D_1, \cdots, D_{i - 1}\rangle$中学习到的知识被最大程度地保留。

### 2.1 解决方法

该框架由三个主要模块组成：数据集成、时空持续表示学习（STCRL）和时空预测（STPrediction），如图1所示。

- **数据集成**：考虑到我们以流的形式将数据输入框架，我们首先从当前数据集$D_i$中采样数据，并通过基于排名的最大干扰检索 ranking-based maximally interfered retrieva（RMIR）采样策略，从存储先前学习观测值子集的重放缓冲区中采样历史数据。然后，在提出的时空混合（STMixup）机制的帮助下，将这两种数据进行集成，目标是更好地积累时空知识并缓解灾难性遗忘。
- **时空持续表示学习（STCRL）**：在这个模块中，采用时空简单孪生（STSimSiam）网络（自监督孪生网络的一种变体[19]），通过互信息最大化进行整体表示学习。为辅助自监督学习，我们基于时空数据的特定特征，提出了五种不同的数据增强方法，即DropNodes（DN）、DeleteEdges（DE）、SubGraph（SG）、AddEdge（AE）和TimeShifting（TS）。然后，将由两种随机选择的数据增强方法生成的增强数据插入到两个相互共享参数的时空编码器（STEncoder）中，接着是一个投影多层感知器（projection MLP，投影器）。
- **时空预测（STPrediction）**：我们使用与STCRL中具有相同参数的STEncoder，从原始数据中学习潜在隐藏特征，学习到的数据随后存储在重放缓冲区中。最后，将学习到的特征输入到时空解码器（STDecoder）中进行预测。

#### 2.1.1 数据集成

![20250321141716](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321141716.png)

为了从重放缓冲区$\mathcal{B}$中采样更具代表性的样本，我们设计了一种新颖的基于排名的最大干扰检索（RMIR）采样方法。给定当前数据序列$D_i$，我们首先从$D_i$中以时间槽$t_i^{k - M + 1}$为起点采样$M$个观测值$\mathcal{X}_M = \langle X_{t_i^{k - M + 1}}, \cdots X_{t_i^{k}}\rangle$，并从重放缓冲区$\mathcal{B}$中选择存储的样本$\mathcal{X}_{\mathcal{B}}$，其中$\mathcal{B}$旨在作为显式记忆，用于保存先前学习的观测值子集（即未经过STMixup的先前训练的观测值）。然后，通过提出的时空混合（STMixup）机制，将所选样本与当前观测值相结合，借助历史观测值来缓解灾难性遗忘。我们可以将数据集成过程公式化如下：
$$
\begin{align}
\mathcal{X}_{\mathcal{B}} &= RMIR(\mathcal{B}, size = |\mathcal{S}|)\\
\mathcal{X}_{mix} &= STMixup(\mathcal{X}_{M}, \mathcal{X}_{\mathcal{B}})
\end{align}
$$
其中$|\mathcal{S}|$是采样大小。

我们继续详细阐述RMIR采样方法和STMixup机制。

**RMIR采样方法**：我们设计了一种RMIR采样方法，从重放缓冲区中选择$|\mathcal{S}|$个有代表性的观测值$\mathcal{X}_{\mathcal{B}}$。通常，现有的基于重放的持续学习方法中的大多数方法是从重放记忆中随机选择观测值，这会导致准确性结果不理想，因为它们可能会忽略用于重放的有代表性的观测值。为了选择更具代表性的样本，我们首先检索$|\mathcal{N}|$个观测值$\mathcal{X}_{\mathcal{N}}$，这些观测值将是受可预见参数更新导致的损失增加影响最大的（即负面冲击最大），其中$|\mathcal{N}|>|\mathcal{S}|$。具体来说，给定$\mathcal{B}$中的观测值$X_{t_i^{\mathcal{B}}}$和标准目标函数$min \mathcal{L}_{RMIR}(f_{\theta}(X_{t_i^{\mathcal{B}}}), Y_{t_i^{\mathcal{B}}})$，其中$f_{\theta}(\cdot)$表示URCL模型，我们通过梯度下降从$X_{t_i^{\mathcal{B}}}$更新参数$\theta$，如公式3所示：

$$
\theta^{v} = \theta - \alpha \nabla \mathcal{L}_{RMIR}(f_{\theta}(X_{t_i^{\mathcal{B}}}), Y_{t_i^{\mathcal{B}}})
$$
其中$\mathcal{L}_{RMIR}$表示采样损失，$f_{\theta}(\cdot)$表示模型，$Y_{t_i^{\mathcal{B}}}$是真实值。注意，平均绝对误差（MAE）被用作采样损失。我们选择前$|\mathcal{N}|$个值。

考虑到时空数据的时间相关性（例如趋势和周期性），由于相似性，很久以前的数据（周期性数据）对当前预测有显著影响。然后，我们通过皮尔逊系数计算$\mathcal{X}_{\mathcal{N}}$和$\mathcal{X}_{M}$中观测值之间的相似性。最后，我们从$\mathcal{X}_{\mathcal{N}}$中采样与$\mathcal{X}_{M}$最相似的前$|\mathcal{S}|$个观测值。通过这种方式，我们不仅可以选择能够缓解灾难性遗忘的样本，还可以选择增强时间依赖性捕捉的样本。

**STMixup机制**：为了利用历史观测值，我们引入STMixup机制，将当前观测值$\mathcal{X}_{M}$和在$\mathcal{B}$中采样的观测值$\mathcal{X}_{\mathcal{B}}$进行融合。具体来说，STMixup在$\mathcal{X}_{M}$和$\mathcal{X}_{\mathcal{B}}$之间进行插值，促使模型在流式时空数据序列中呈现线性行为，以最小化灾难性遗忘。

通常，假设$(x_i, y_i)$和$(x_j, y_j)$是训练数据中两个随机选择的特征 - 目标对，STMixup基于邻域风险最小化原则[47]通过插值生成虚拟训练示例，以扩大训练分布的支持范围，从而克服概念漂移。在STMixup中，我们使用观测值 - 真实值对来表示训练中的特征 - 目标对。我们使用$(\tilde{x}, \tilde{y})$表示原始两对附近的插值特征 - 目标对。
$$
\begin{align}
\tilde{x} &= \lambda \cdot x_i + (1 - \lambda) \cdot x_j\\
\tilde{y} &= \lambda \cdot y_i + (1 - \lambda) \cdot y_j
\end{align}
$$
其中$\lambda \sim Beta(\alpha, \alpha)$，且$\alpha \in (0, \infty)$。我们通过在当前观测值$\mathcal{X}_{M}$和通过RMIR采样从重放缓冲区$\mathcal{B}$中采样的观测值$\mathcal{X}_{\mathcal{B}}$之间进行插值来设计STMixup。STMixup后的插值观测值$\mathcal{X}_{mix}$公式如下：
$$
\mathcal{X}_{mix} = \lambda \cdot \mathcal{X}_{M} + (1 - \lambda) \cdot \mathcal{X}_{\mathcal{B}}
$$
插值观测值$\mathcal{X}_{mix}$可以通过回顾重放缓冲区$\mathcal{B}$中的过去实例，增强模型的持续学习能力，这些实例会受到可预见参数的最大负面影响。此外，STMixup可以引入正则化损失最小化的近似[48]，以避免过拟合。

#### 2.1.2 时空持续表示学习（STCRL）

![20250321141716](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321141716.png)

插值观测值$\mathcal{X}_{mix}$被输入到时空持续表示学习（STCRL）模块中进行整体特征学习。STCRL是一个精心设计的自监督学习模块，它由两部分组成：时空数据增强和STSimSiam网络。为了实现有效的时空学习，我们考虑时空数据的独特属性，提出了五种定制的时空数据增强方法，将样本（即传感器网络中的观测值）$\mathcal{G} = [\mathcal{X}_{mix}; G]$转换为其相应的扰动$\mathcal{G}'$。我们随机选择两个不同的扰动$\mathcal{G}_1'$和$\mathcal{G}_2'$，然后将它们输入到一个STSimSiam网络中，该网络由两个STEncoder $f_{\theta_{STE}}$和一个投影多层感知器头$h(\cdot)$组成。STSimSiam的目的是更好地捕捉时空依赖性。最后，STSimSiam最大化两个选定扰动表示之间的互信息，以确保整体特征保留。

我们继续详细阐述时空数据增强和STSimSiam网络。

**时空数据增强**：我们使用五种精心设计的时空增强方法对插值观测值$\mathcal{X}_{mix}$进行增强，这些方法构建语义相似的样本对，并通过应对扰动来提高学习表示的质量。尽管现有研究针对图数据提出了几种数据增强方法[49]-[51]，但由于时空数据具有复杂的时空相关性（例如临近性），这些方法对时空数据效果不佳。因此，我们提出了五种基于空间的数据增强方法（即DropNodes (DN)、DropEdge (DE)、SubGraph (SG) 和AddEdge (AE) ）以及一种基于时间的方法（即TimeShifting (TS) ）。下面介绍每种方法：

![20250321205122](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321205122.png)

- **DN（DropNodes）**：如图2(a)所示，给定样本$\mathcal{G} = [\mathcal{X}_{mix}; G]$，DN按照一定的分布（例如均匀分布）随机丢弃$G$中一定比例（例如10%）的节点，以得到$\mathcal{G}' = [\mathcal{X}_{mix}; G']$，这确保了缺失的节点对$G$的语义（例如分布）没有影响。具体来说，我们将邻接矩阵$A$中与丢弃节点对应的元素进行掩码，以扰动图结构。
    $$
    A_{i,j}' =
    \begin{cases}
    0, & \text{如果 } v_i \text{ 被丢弃} \\
    A_{i,j}, & \text{否则}
    \end{cases}
    $$
    理想情况下，DN可以通过减少受传感器或通信故障等导致的缺失数据的影响，来提高模型的鲁棒性。
- **DE（DropEdge）**：DE随机删除部分边，如图2(b)所示。由于传感器网络$G$的权重对于描述节点之间的空间相关性（例如距离和相似性）很重要。我们首先按照特定分布从$G$中采样一定比例的边$\mathcal{E}$，然后设置一个阈值$\theta_{DE}$。如果$\mathcal{E}$中边的权重低于$\theta_{DE}$，我们就删除相应的边，公式如下：
    $$
    a_{i,j}' =
    \begin{cases}
    0, & \text{如果 } a_{i,j} < \theta_{DE} \\
    a_{i,j}, & \text{否则}
    \end{cases}
    $$
    其中$a_{i,j}$表示节点$v_i$和$v_j$之间边的权重，$a_{i,j}'$是更新后的权重。设置阈值的目的是保留边的重要连接性。
- **SG（SubGraph）**：SG通过随机游走从$G = (V, E)$中采样一个子图$G' = (V', E')$，以最大程度地保留传感器网络的语义（见图2(c)），其中$V' \subseteq V$且$E' \subseteq E$。通过SG，我们旨在通过对子图进行特征学习来改善局部空间相关性的捕捉。
- **AE（AddEdge）**：AE随机选择一定比例的远距离节点对（例如，超过三跳），并在每对节点之间添加边，如图2(d)所示。这些添加边的权重设置为相应节点对的点积相似度。考虑节点对$(v_i, v_j)$，相应的权重$w_{i,j}$计算如下：
    $$w_{i,j} = \vec{\mathcal{X}}_{mix}^i \cdot \vec{\mathcal{X}}_{mix}^j$$
    其中$\vec{\mathcal{X}}_{mix}^i$是表示节点$v_i$特征的向量。AE的目的是通过连接彼此相似的远距离节点对，增强我们的模型捕捉全局空间相关性的能力。

- **TS（TimeShifting）**：如图2(e)所示，TS包括时间切片、时间扭曲和时间翻转，在时域中对$\mathcal{G}$中的当前观测值$\mathcal{X}_{mix}$进行转换。注意，我们在模型训练时随机选择TS中的一种方法。

  - **Time Slicing（时间切片）**：时间切片通过随机提取长度为$l$的连续切片$\mathcal{X}_{mix}^{slice}$，在时域中对当前观测值$\mathcal{X}_{mix}$进行子采样。正式地：
    $$\mathcal{X}_{mix}^{slice} = \langle X_{mix,t_i^{s - l + 1}}, \cdots, X_{mix,t_i^{s}} \rangle$$
    其中$t_i^{k - M + 1} \leq t_i^{s - l + 1} \leq t_i^{s} \leq t_i^{k}$，$t_i^{k}$是当前时间槽，$M$是观测值的长度，$X_{mix,t_i^{s}}$表示在时间槽$t_i^{s}$经过STMixup后的观测值。

  - **Time Warping（时间扭曲）**：时间扭曲通过线性插值对切片观测值$\mathcal{X}_{mix}^{slice}$进行上采样，生成扭曲后的观测值$\mathcal{X}_{mix}^{warp}$，其长度与$\mathcal{X}_{mix}$相等：
    $$\mathcal{X}_{mix}^{warp} = \langle X_{mix,t_i^{k - M + 1}}', \cdots, X_{mix,t_i^{k}}' \rangle$$
    其中$X_{mix,t_i^{k}}'$表示在时间槽$t_i^{k}$生成的插值观测值。

  - **Time Flipping（时间翻转）**：时间翻转$TF(\cdot)$翻转扭曲后观测$\mathcal{X}_{mix}^{warp}$的时序，以在时域中生成一个新的序列$\mathcal{X}_{mix}^{flip}$：
    $$\mathcal{X}_{mix}^{flip} = \langle X_{mix,t_i^{k}}', \cdots, X_{mix,t_i^{k - M + 1}}' \rangle$$

    我们对通过STMixup生成的集成观测值$\mathcal{X}_{mix}$随机应用两种不同的数据增强方法，以获得两个增强观测值$\mathcal{X}_{mix}^{aug1}$和$\mathcal{X}_{mix}^{aug2}$。

![20250321141716](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321141716.png)

**STSimSiam网络**：受到自监督学习强大的特征学习能力的启发，我们在自监督学习的指导下设计了一种新颖的STSimSiam网络，用于捕捉整体的时空表示，输入两个随机增强的观测值$\mathcal{X}_{mix}^{aug1}$和$\mathcal{X}_{mix}^{aug2}$。更具体地说，STSimSiam由两个STEncoder组成，分别用于学习$\mathcal{X}_{mix}^{aug1}$和$\mathcal{X}_{mix}^{aug2}$的时空表示，以及一个投影头，用于将$\mathcal{X}_{mix}^{aug1}$的潜在嵌入投影到$\mathcal{X}_{mix}^{aug2}$的潜在空间中。最后，考虑到互信息最大化已被证明可以学习整体特征，从而改善持续学习[18]，我们通过GraphCL损失最大化从$\mathcal{X}_{mix}^{aug1}$和$\mathcal{X}_{mix}^{aug2}$学习到的表示之间的互信息。

如图1右上角所示，我们将两个随机增强的观测值$\mathcal{X}_{mix}^{aug1}$和$\mathcal{X}_{mix}^{aug2}$输入到STSimSiam网络中。我们首先通过由特定时空网络（在本文中为GraphWaveNet）组成的STEncoder $f_{\theta_E}$将两个增强观测值编码为两个固定向量$z_1$和$z_2$，以捕捉复杂的时空依赖性。接下来，我们将$z_1$输入到一个投影多层感知器头中，将其映射到$z_2$的潜在空间中，该投影头包含几个表示为$h(\cdot)$的多层感知器层。这个过程可以公式化如下：
$$z_1 = f_{\theta_E}(\mathcal{X}_{mix}^{aug1}), \ p_1 = h(z_1), \ z_2 = f_{\theta_E}(\mathcal{X}_{mix}^{aug2})$$
我们使用stopgrad操作$SG(\cdot)$[52]来防止STSimSiam得到平凡解。例如，$SG(z_2)$表示$\mathcal{X}_{mix}^{aug2}$上的STEncoder不会从$z_2$接收梯度。

为了增强整体特征保留，我们使用互信息最大化来最大化$\mathcal{X}_{mix}^{aug1}$和$\mathcal{X}_{mix}^{aug2}$表示之间的相似性。首先，对于当前的两个增强观测值，我们使用余弦相似度来测量它们的输出向量$p_1$和$z_2$之间的相似度$\mathcal{C}(\cdot)$，公式如下：
$$
\begin{align*}
\mathcal{C}(p_1, z_2) &= \frac{p_1}{\|p_1\|_2} \cdot \frac{SG(z_2)}{\|SG(z_2)\|_2}\\
&= \frac{h(f_{\theta}(\mathcal{X}_{mix}^{aug1}))}{\|h(f_{\theta}(\mathcal{X}_{mix}^{aug1}))\|_2} \cdot \frac{SG(f_{\theta}(\mathcal{X}_{mix}^{aug2}))}{\|SG(f_{\theta}(\mathcal{X}_{mix}^{aug2}))\|_2}
\end{align*}
$$
其中$\|\cdot\|_2$是$l_2$范数。

我们用$(\mathcal{X}_{mix}^{aug1}, \mathcal{X}_{mix}^{aug2})$表示一个增强观测值对。对于一个包含$\mathcal{S}$个增强观测值对的小批量数据，我们采用GraphCL[49]损失来最大化它们的互信息。第$s$个增强观测值对的GraphCL损失$L_{ssl}^{s}$定义如下：
$$L_{ssl}^{s} = -log\frac{exp(\mathcal{C}(p_{s,1}, z_{s,2}) / \tau)}{\sum_{s' = 1, s' \neq s}^{\mathcal{S}} exp(\mathcal{C}(p_{s,1}, z_{s',2}) / \tau)}$$
其中$p_{s,1}$和$z_{s,2}$表示第$s$个增强观测值对$(\mathcal{X}_{mix,s}^{aug1}, \mathcal{X}_{mix,s}^{aug2})$的输出向量，$\mathcal{C}(\cdot)$是余弦相似度，$\tau$是温度参数。为了提取更有效的特征[19]，我们定义一个对称相似度函数并得到最终的$L_{ssl}^{s}$：
$$L_{ssl}^{s} = -log\frac{exp((\frac{1}{2}\mathcal{C}(p_{s,1}, z_{s,2}) + \frac{1}{2}\mathcal{C}(p_{s,2}, z_{s,1})) / \tau)}{\sum_{s' = 1, s' \neq s}^{\mathcal{S}} exp((\frac{1}{2}\mathcal{C}((p_{s,1}, z_{s',2}) + \frac{1}{2}\mathcal{C}(p_{s,2}, z_{s',1})) / \tau)}$$
其中$p_{s,2} = h(f_{\theta_E}(\mathcal{X}_{mix,s}^{aug2}))$且$z_{s,1} = f_{\theta_E}(\mathcal{X}_{mix,s}^{aug1})$。

最终的GraphCL损失是在小批量中的所有增强对之间计算的，如下所示：
$$L_{ssl} = \frac{1}{\mathcal{S}} \sum_{s = 1}^{\mathcal{S}} L_{ssl}^{s}$$
其中$\mathcal{S}$是批量大小。

#### 2.1.3 时空预测（STPrediction）

我们设计了一个时空预测网络，包括一个时空编码器（STEncoder）和一个时空解码器（STDecoder）。具体来说，我们将插值观测值$\mathcal{X}_{mix}$输入到STEncoder $f_{\theta_E}(\cdot)$中，通过捕捉复杂的时空依赖性来学习高维表示。该STEncoder与STSimSiam中的STEncoder共享参数。学习到的表示被输入到STDecoder $f_{\theta_D}$中进行预测。同时，最近学习的数据存储在重放缓冲区中。这个过程可以公式化如下：
$$h_{\theta} = f_{\theta_E}(\mathcal{X}_{mix}), \ \hat{\mathcal{Y}} = f_{\theta_D}(h_{\theta})$$
其中$\hat{\mathcal{Y}}$是预测结果。

我们的框架的优点之一是它的通用性。它可以很容易地作为插件应用于大多数现有的遵循自动编码器架构的时空预测模型。GraphWaveNet[9]是一种能够捕捉精确空间依赖性和长期时间依赖性的先进时空预测模型，但它不是在自动编码器架构下开发的。受其在深度时空图建模上的出色性能启发，我们在工作中以它为例，展示如何重新组织其架构以符合自动编码器架构（即STEncoder和STDecoder）。具体来说，在STEncoder中，如图3所示，使用与门控时间卷积层集成的图卷积层来捕捉时空依赖性；而在STDecoder中，如图4所示，应用几个前馈网络将高维特征映射为低维输出用于预测。特别值得注意的是，我们在5.2.4节的实验部分研究了不同时空预测模型（包括基于RNN的DCRNN[4]和基于注意力的GeoMAN[53]）的效果。研究表明，我们的框架可以适应不同的预测模型。对于现有的缺少STDecoder的时空预测网络，我们使用堆叠的多层感知器（MLP）作为STDecoder。

![20250321212110](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321212110.png)

**STEncoder**：STEncoder的架构如图3所示。以$\mathcal{X}_{mix}$作为输入，一个MLP层将其映射到高维潜在空间，然后将学习到的特征输入到门控时间卷积网络（TCN）层，以学习输入观测值之间的时间相关性。接下来，使用图卷积网络（GCN）层捕捉观测值之间的空间相关性，其中使用残差操作以确保准确性。第$i$个时空层学习到的特征可以公式化如下：
$$h_{\theta}^{i} = f_{\mathcal{G}}(GatedTCN(W_{\theta}^{i} \cdot h_{\theta}^{i - 1} + b_{\theta}^{i}), A)$$
其中$f_{\mathcal{G}}(\cdot)$表示GCN，$A$是邻接矩阵，$W_{\theta}^{i}$是可学习参数，$b_{\theta}^{i}$表示偏差，$h_{\theta}^{0}$等于$\mathcal{X}_{mix}$。门控TCN层由两个并行的TCN层（即$TCN_a$和$TCN_b$）组成。

**图卷积层**：最近的研究非常关注将卷积网络推广到图数据。在这项工作中，我们在构建的传感器网络上使用谱卷积，其可以简单地公式化如下：
        $$f_{\mathcal{G}}(X, A) = \sigma(\widetilde{A}XW^{t})$$
        其中$f_{\mathcal{G}}$表示GCN操作，$X$表示节点特征，$\widetilde{A} = A + I_N$是在归一化后添加自连接的$\mathcal{G}$的邻接矩阵，$W^{t}$表示可学习的权重矩阵，$\sigma(\cdot)$是激活函数。

基于地理学第一定律：“近的事物比远的事物相关性更强”[1]，我们首先通过考虑节点之间的地理距离来构建局部空间图。如果两个节点$v_i$和$v_j$在地理上相互连接，则它们之间存在一条边，并且相应的权重设置如下：
$$
        A_{i,j} =
        \begin{cases}
        \frac{1}{dis}, & \text{如果 } v_i \text{ 连接到 } v_j \\
        0, & \text{否则}
        \end{cases}
$$

其中$dis$表示两个节点$v_i$和$v_j$之间的地理距离。遵循采用扩散GCN的扩散卷积递归神经网络（DCRNN）[4]，我们通过对图信号的$K$有限步扩散过程进行建模，将扩散卷积层推广为公式19的形式，如公式21所示：
$$f_{\mathcal{G}}(X, A) = \sigma(\sum_{k = 0}^{K} P_{k}XW_{k})$$
其中$P_{k}$表示转移矩阵的幂级数。对于无向图，我们可以得到$P = \widetilde{A}/rowsum(\widetilde{A})$；而对于有向图，扩散过程有两个方向：前向$P^{f} = \widetilde{A}/rowsum(\widetilde{A})$和后向$P^{b} = \widetilde{A}^{T}/rowsum(\widetilde{A}^{T})$。有向图的扩散图卷积层推导如下：
$$
    f_{\mathcal{G}}(X, A) = \sigma(\sum_{k = 0}^{K} P_{k}^{f}XW_{k_1} + P_{k}^{b}XW_{k_2})
$$

然而，地理学第一定律可能无法完全反映城市区域的空间相关性，特别是全局空间相关性（例如，兴趣点相似性）[2, 6]。为了解决这个问题，我们通过将两个随机初始化的具有可学习参数$E_1$和$E_2$的节点嵌入相乘来构建自适应邻接矩阵$\widetilde{A}_{adp}$：
$$\widetilde{A}^{adp} = Softmax(ReLU(E_1E_2^{T}))$$
其中Softmax用于对自适应邻接矩阵进行归一化。通过综合考虑局部和全局空间相关性，最终的图卷积层由公式24给出：
$$
    f_{\mathcal{G}}(X, A) = \sigma(\sum_{k = 0}^{K} P_{k}^{f}XW_{k_1} + P_{k}^{b}XW_{k_2} + \widetilde{A}_{k}^{adp}XW_{k_3})
$$
如果传感器网络未知，我们仅采用自适应邻接矩阵来捕捉空间依赖性。

**门控卷积层**：为了捕捉时间依赖性，我们采用扩张因果卷积[54]作为我们的TCN，因为它具有对长期时间相关性进行建模和并行计算的能力。具体来说，给定数据序列$\mathcal{X}$和滤波器$f$，$x$在第$j$步的扩张因果卷积操作表示为：

$$
    \mathcal{X} \circ f(j) = \sum_{m = 0}^{K - 1} f(m)\mathcal{X}(j - d \times m)
$$
其中$d$表示反映跳跃步数的扩张因子，$K$是滤波器$f$的长度。

门控机制已被证明对控制TCN层间的信息流很有用[55]。在我们的工作中，我们采用一种简单形式的门控TCN来对复杂的时间相关性进行建模，其中门控TCN仅包含一个输出门。给定观测值$\mathcal{X}$，门控TCN如公式26所示：
$$
    h = g(W_1 \times \mathcal{X} + b) \odot \sigma(W_2 \times \mathcal{X} + c)
$$
其中$h$是由第一个MLP层建模的输入的学习特征，$W_1$和$W_2$是可学习参数，$b$和$c$是偏差，$\odot$表示元素级乘积，$g(\cdot)$和$\sigma(\cdot)$表示激活函数（例如，tanh和sigmoid）。

**STDecoder**：然后将学习到的特征输入到STDecoder中，对数据表示进行解码以进行预测。如图4所示，STDecoder包含几个堆叠的前馈层（即MLP），随后是激活函数（即ReLU），以学习用于未来预测的投影函数。它可以公式化如下：
$$\hat{\mathcal{Y}} = W_{\theta_D}(\alpha(h_{\theta}) + b_{\theta_D})$$
其中$\hat{\mathcal{Y}}$表示预测结果，$W_{\theta_D}$是可学习参数，$\alpha(\cdot)$是ReLU激活函数，$b_{\theta_D}$是偏差。

**整体目标函数**
URCL的整体目标是通过平均绝对误差（MAE）来最小化每个数据流$D_i$的预测误差。目标函数如下：
$$L_{task} = \frac{1}{\mathcal{L}} \sum_{l = 1}^{\mathcal{L}} |\hat{\mathcal{Y}}^{l} - \mathcal{Y}^{l}|$$
其中$\mathcal{L}$是训练样本大小，$\hat{\mathcal{Y}}^{l}$是预测值，$\mathcal{Y}^{l}$是真实值。

最终的损失包含两部分：预测任务的预测损失$L_{task}$和用于整体特征学习的GraphCL损失$L_{ssl}$。我们将它们结合起来，总体损失如下：
$$L_{all} = L_{task} + L_{ssl}$$

URCL的整个过程如算法1所示，其中第2 - 3行说明了数据预处理，第4 - 12行展示了URCL的训练过程。

### 2.2 模型结构

![20250321141716](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321141716.png)

## 三、实验验证与结果分析

### 3.0 实验设置

**持续学习设置**：我们将$D_1$表示为基础集，$D_2$到$D_m$表示为增量集，其中$\mathbb{D} = \langle D_1, D_2, \cdots, D_m\rangle (m \geq 1)$是如第三节所定义的流式时空数据序列。在实验中，我们使用一个基础集，记为$\mathcal{B}_{set}$，以及四个增量集，记为$\mathcal{I}_{set}^1$到$\mathcal{I}_{set}^4$，以方便进行持续训练。具体来说，我们将每个数据集的30%作为基础集，将每个数据集中剩余的数据等分为四个部分以形成增量集。基础集和增量集随着时间的推移依次提供。

**训练过程**：我们试图通过将基线方法的原始训练过程映射到一个持续训练过程，在基线方法之间建立公平的比较（见第五节A2部分），如图5所示。我们首先在基础集$D_1$上训练一个初始的时空（ST）模型，然后基于最后学习到的模型，在增量集（$D_2 - D_5$）上重新训练一个新的ST模型。

### 3.1 性能实验

![20250321224433](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321224433.png)

1. **流式数据训练性能**：我们的URCL提出了一种基于重放的流式数据训练策略，利用重放缓冲区存储先前学习的样本，并通过时空混合（STMixup）机制与训练数据融合，从而有效保留历史知识。为评估在流式数据上的训练性能，采用两种具有代表性的训练策略替代URCL中的基于重放的策略，并与URCL进行比较：
    - **OneFitAll**：使用基础集训练模型，然后对时空数据集流中的所有测试数据进行预测；
    - **FinetuneST**：在基础集上训练初始模型，然后使用增量集反复微调模型。

    以GraphWaveNet作为OneFitAll和FinetuneST的基础模型。在表II中报告了在PEMS - BAY和PEMS08数据集上的平均绝对误差（MAE）和均方根误差（RMSE）结果，其中整体最佳性能以粗体标记。为节省空间，未报告METR - LA和PEMS04数据集的结果，因为它们与PEMS - BAY和PEMS08的结果相似，其他数据集的结果包含在技术报告中。

    结果表明，在两个数据集上，与其他方法相比，URCL在MAE和RMSE方面均表现最佳。具体而言，在PEMS - BAY数据集上，URCL的MAE和RMSE分别比最佳基线方法高出14.5% - 67.3%和15.5% - 72.4%；在PEMS08数据集上，MAE和RMSE分别高出1.2% - 49.1%和8.9% - 35.5%。OneFitAll和FinetuneST在两个数据集的基础集上能提供可接受的MAE和RMSE结果，但在增量集上性能会下降。OneFitAll表明随着时间推移，新数据与训练数据可能不同，即出现概念漂移，静态模型不再有效，需要持续学习（CL）模型。而简单基于CL的FinetuneST并不够，因为它存在遗忘问题。URCL在基础集和增量集上均实现了相对稳定的性能，证明了其优越性。

![20250321224604](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321224604.png)
2. **整体准确性**：在表III中报告了各方法的MAE和RMSE值。现有方法（ARIMA、DCRNN、STGCN、MTGNN、AGCRN和STGODE）的最佳性能以下划线标记，整体最佳性能以粗体标记。注意，为了与基于重放的策略进行公平比较，我们在每个基础集和增量集上对每个原始基线进行重复训练（如图5所示），观察结果如下：
    - URCL在大多数情况下在所有基线中取得最佳结果，MAE和RMSE分别比基线中的最佳结果高出36.0%和34.1%。在大多数情况下，URCL在交通速度数据集（即METR - LA和PEMS - BAY）上的MAE和RMSE方面优于基线中的最佳方法，但METR - LA上的增量集$\mathcal{I}_{set}^4$除外。此外，URCL在交通流量数据集（即PEMS04和PEMS08）上的MAE和RMSE方面也优于基线中的最佳方法，但PEMS08上的增量集$\mathcal{I}_{set}^1$除外。观察发现，URCL在交通速度数据集上的性能提升超过了在交通流量数据集上的提升，这是因为交通流量数据比交通速度数据多得多，因此，使用基于重放的训练策略且有足够数据的基线可以得到可比的结果。
3. **不同主干网络的影响**：接下来研究使用不同主干网络（基础模型）的影响。例如，URCL的主干网络是基于CNN的GraphWaveNet（见IV - D节）。除了基于CNN的模型外，现有基于图的时空预测模型主要有两大流派，包括基于RNN的模型和基于注意力的模型，它们分别采用RNN和注意力机制来学习时间动态，并使用图神经网络捕捉空间相关性。选择两个有代表性的模型作为URCL的主干网络，即基于RNN的DCRNN和基于注意力机制的GeoMAN，然后将它们与URCL进行比较。为简单起见，使用主干网络的名称作为比较方法的名称。表IV显示了在METR - LA和PEMS04上的预测结果。DCRNN和GeoMAN在两个数据集上的MAE和RMSE方面不相上下。URCL在大多数情况下性能最佳，但其他两个模型的性能也具有可比性，特别是在PEMS04上，这证明了URCL采用不同主干网络的通用性。

### 3.2 消融实验

![20250321224754](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321224754.png)

- **w/o STMixup (w/o_STU)**：无STMixup模块的URCL模型，直接连接原始观测值和从重放缓冲区采样的观测值。
- **w/o RMIR Sampling (w/o_RMIR)**：用随机采样机制取代URCL中的RMIR采样机制。
- **w/o STAugmentation (w/o_STA)**：无时空数据增强的URCL模型。
- **w/o GraphCL (w/o_GCL)**：无GraphCL损失的URCL模型。

![20250321224904](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250321224904.png)

**不同主干网络的影响**：接下来研究使用不同主干网络（基础模型）的影响。例如，URCL的主干网络是基于CNN的GraphWaveNet（见IV - D节）。除了基于CNN的模型外，现有基于图的时空预测模型主要有两大流派，包括基于RNN的模型和基于注意力的模型，它们分别采用RNN和注意力机制来学习时间动态，并使用图神经网络捕捉空间相关性。选择两个有代表性的模型作为URCL的主干网络，即基于RNN的DCRNN和基于注意力机制的GeoMAN，然后将它们与URCL进行比较。为简单起见，使用主干网络的名称作为比较方法的名称。表IV显示了在METR - LA和PEMS04上的预测结果。DCRNN和GeoMAN在两个数据集上的MAE和RMSE方面不相上下。URCL在大多数情况下性能最佳，但其他两个模型的性能也具有可比性，特别是在PEMS04上，这证明了URCL采用不同主干网络的通用性。
