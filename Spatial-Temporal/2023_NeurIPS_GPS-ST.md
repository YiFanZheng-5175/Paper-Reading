# GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks

![20250313153746](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313153746.png)
>领域：时空序列预测  
>发表在：NeurIPS 2023  
>模型名字：***G***enerative ***P***re-***T***raining of ***S***patio-***T***emporal  
>文章链接：[GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html)  
>代码仓库：[https://github.com/HKUDS/GPT-ST](https://github.com/HKUDS/GPT-ST)  
![20250313153115](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313153115.png)

## 一、研究背景与问题提出

### 1.1 研究现状

尽管现有方法在时空预测中取得了显著成果，但仍有几个问题未得到充分解决。

- i）***缺乏特定时空模式的定制表示***。定制可分为两个关键方面：时间动态以及时间和空间域中的节点特定属性。时间动态模式在不同时间段表现出变化，例如同一区域周末和工作日之间的对比模式。此外，区域之间的相关性会随着时间动态演变，这超出了静态图表示的能力。特定节点的模式突出了不同区域中观察到的不同时间序列，而非共同模式。***另外，重要的是要确保不同区域在消息聚合后仍保留其各自的节点特征，以防止空间域中突出节点的干扰***。我们认为，对所有这些定制特征进行编码对于保证模型的稳健性至关重要。然而，现有工作往往在全面考虑这些因素方面有所不足。
![20250313153852](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313153852.png)
- ii）***对不同层次的空间依赖性考虑不足***。大多数方法在对空间依赖性进行建模时主要关注区域之间的成对关联，但它们忽略了不同空间层次的语义相关性。在现实世界中，具有相似功能的区域往往表现出相似的时空模式。通过对不同区域进行聚类分析，模型可以探索相似区域之间的共同特征，从而促进更好的空间表示学习。此外，当前研究中缺乏对不同类型高级区域之间的高层区域关系进行充分建模。不同类型的高级区域之间的时空模式可能表现出动态转移关系。例如，在工作时间，有明显的人流从居住区向工作区移动，如图 1 右侧部分所示。在这种情况下，居住区内的人流变化可以为预测工作区内的人流提供有价值的辅助信号。这突出了纳入不同层次区域之间的细粒度和粗粒度相关性对于增强时空预测模型的预测能力的重要性。

- iii）***以往的在图上的预训练任务没考虑复杂的时间演变模式和空间相关机制***，到近年来，图数据的自监督学习方法受到了广泛关注 [53,38]。基于对比学习的图神经网络通过数据增强技术生成原始图的不同视图 [42,29]。然后使用损失函数在视图之间最大化正样本对的一致性，同时最小化负样本对的一致性。例如，GraphCL [47] 通过应用节点丢弃和边打乱生成图的两个视图，并在它们之间进行对比学习。另一个研究方向集中在生成式图神经网络上，其中图数据本身通过重建作为表示学习的自然监督信号。GPTGNN [17] 通过重建图特征和边进行预训练，而 GraphMAE [16] 在图编码器和解码器中都使用节点特征掩码来重建特征并学习图表示。然而，时空预测任务需要同时考虑复杂的时间演变模式和空间相关机制。专门为这类任务设计的预训练范式仍然是一个探索和研究的领域。

## 二、问题剖析与解决策略

### 2.1 解决方法

一种解决上述挑战的直观方法是开发端到端模型。然而，当前的模型在从这种方法中受益方面面临困难，因为最先进（SOTA）模型中的每个模块都经过了复杂的优化，任何拆卸和整合都可能导致预测性能下降。那么，是否有一种策略可以让现有的时空方法利用端到端模型的优势呢？答案是肯定的。最近，自然语言处理（NLP）领域的 ChatGPT [3, 27] 和计算机视觉（CV）领域的 MAE [15] 等预训练框架已由先驱者提出，并在各自领域得到了广泛研究。这些预训练架构构建无监督训练任务，包括掩码或其他技术，以学习更好的表示并提高下游任务性能。然而，到目前为止，这种可扩展的预训练模型在时空预测领域的应用还很有限。为了解决上述挑战，我们提出了一种名为时空预测生成式预训练（GPT-ST）的新颖框架。该框架旨在无缝集成到现有的时空预测模型中，提高其性能。我们的贡献总结如下

- 我们提出了GPT - ST，这是一种专为时空预测设计的新型预训练框架。该框架可无缝集成到现有的时空神经网络中，从而提升性能。我们的方法将模型参数定制方案与自监督掩码自动编码相结合，实现了有效的时空预训练。
- GPT - ST巧妙地利用层次超图结构，从***全局角度***捕捉***不同层次***的空间依赖关系。通过与精心设计的自适应掩码策略协同工作，该模型能够对区域间的***簇内和簇间***空间关系进行建模，从而生成稳健的时空表示。
- 我们在真实世界的数据集上进行了大量实验，不同下游基线性能的显著提升证明了GPT - ST的优越性能。

***(说实话下面这些东西完全看不懂在做什么)***

#### 2.1.1 时空预训练范式

我们的GPT - ST框架旨在开发一种预训练的时空（ST）表示方法，以提高下游时空预测任务（如交通流量预测）的准确性。如图2所示，GPT - ST的工作流程可分为预训练阶段和下游任务阶段。我们将这两个阶段表述如下：

GPT - ST框架的预训练阶段采用掩码自动编码（MAE）任务作为训练目标。此MAE任务的目标是通过学习一个时空表示函数$f$，根据未掩码的信息来重建时空数据的掩码信息。对于时间段$[t_{K - L + 1}, t_{K}]$内的训练时空数据，我们的目标是最小化以下目标函数：

$$
\mathcal{L}((1 - \mathbf{M}) \odot \mathbf{X}_{t_{K - L + 1}:t_{K}}, \mathbf{W} \cdot f(\mathbf{M} \odot \mathbf{X}_{t_{K - L + 1}:t_{K}})) \tag{1}
$$

其中，$\mathcal{L}$表示预测偏差的度量，$\mathbf{M} \in \{0, 1\}$表示掩码张量，$\odot$表示逐元素乘法运算，$\mathbf{W}$表示一个预测线性层。

在预训练阶段之后，GPT - ST的结果将用于下游任务阶段。预训练模型生成高质量的时空表示，以辅助诸如交通预测等下游预测任务。具体而言，下游阶段表述为：

$$
\zeta = f(\mathbf{X}_{t_{K - L + 1}:t_{K}}); \quad \hat{\mathbf{X}}_{t_{K + 1}:t_{K + P}} = g(\zeta, \mathbf{X}_{t_{K - L + 1}:t_{K}}) \tag{2}
$$
其中，$\zeta$是函数$f$根据第$K$个时间步之前$L$个时间步的历史时空数据生成的表示。输出是对接下来$P$个时间步的预测。各种现有的时空神经网络都可以作为预测函数$g$。

#### 2.1.2 定制的时间模式编码

![20250313162722](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313162722.png)

##### 初始嵌入层

我们首先构建一个嵌入层，用于初始化时空数据$\mathbf{X}$的表示。原始数据使用Z - Score函数[1, 39]进行归一化处理，然后通过掩码操作进行掩码。随后，应用线性变换将表示增强为
$$
\mathbf{E}_{r,t} = \mathbf{M}_{r,t} \odot \tilde{\mathbf{X}}_{r,t} \cdot \mathbf{E}_{0}
$$
$\mathbf{E}_{r,t} \in \mathbb{R}^{d}$表示第$t$个时间步中第$r$个区域的表示  
$\mathbf{M}_{r,t} \in \mathbb{R}^{F}$表示掩码操作  
$\tilde{\mathbf{X}}_{r,t} \in \mathbb{R}^{F}$表示归一化后的时空数据。  
$d$表示隐藏单元的数量。  
$\mathbf{E}_{0} \in \mathbb{R}^{F \times d}$表示$F$个特征维度的可学习嵌入向量。

##### 时间超图神经网络

为了促进全局关系学习，我们使用超图神经网络进行时间模式编码，具体如下：
$$
\boldsymbol{\Gamma}_{t} = \sigma(\overline{\mathbf{E}}_{t} \cdot \mathbf{W}_{t} + \mathbf{b}_{t} + \mathbf{E}_{t}); \overline{\mathbf{E}}_{r} = \text{HyperPropagate}(\mathbf{E}_{r}) = \sigma(\mathbf{H}_{r}^{\top} \cdot \sigma(\mathbf{H}_{r} \cdot \mathbf{E}_{r})) \tag{3}
$$
$\boldsymbol{\Gamma}_{t}$表示第$t$个时间步的结果  
$\overline{\mathbf{E}}_{t}$表示中间嵌入  
$\mathbf{E}_{t} \in \mathbb{R}^{R \times d}$表示初始区域嵌入。  
$\mathbf{W}_{t} \in \mathbb{R}^{d \times d}$，$\mathbf{b}_{t} \in \mathbb{R}^{d}$表示第$t$个时间步特定的参数。  
$\sigma(\cdot)$表示LeakyReLU激活函数。  
中间嵌入$\overline{\mathbf{E}} \in \mathbb{R}^{R \times T \times d}$通过超图信息传播计算得出。  
它使用区域特定的超图$\mathbf{H}_{r} \in \mathbb{R}^{H_{T} \times T}$沿着时间步之间的连接和$H_{T}$条超边传播第$r$个区域的时间嵌入$\mathbf{E}_{r} \in \mathbb{R}^{T \times d}$，以便捕捉时间段之间的多步关系。

人话：就是2024年HimNet的元参数学习那部分，本质上是个图卷积，然后参数会乘上time emb和node emb

##### 定制参数学习器

为了描述时间模式的多样性，我们的时间编码器针对不同区域和不同时间段进行模型参数定制。具体来说，前面提到的时间特定参数$\mathbf{W}_{t}$、$\mathbf{b}_{t}$以及区域特定的超图参数$\mathbf{H}_{r}$是通过可学习的过程生成的，而不是直接使用独立参数。具体而言，定制参数的学习方式如下：
$$
\mathbf{H}_{r} = \mathbf{c}_{r}^{\top} \overline{\mathbf{H}}; \mathbf{W}_{t} = \mathbf{d}_{t}^{\top} \overline{\mathbf{W}}; \mathbf{b}_{t} = \mathbf{d}_{t}^{\top} \overline{\mathbf{b}}; \mathbf{d}_{t} = \text{MLP}(\overline{\mathbf{z}}_{t}^{(d)} \mathbf{e}_{1} + \overline{\mathbf{z}}_{t}^{(w)} \mathbf{e}_{2}) \tag{4}
$$
其中，$\overline{\mathbf{H}} \in \mathbb{R}^{d^{\prime} \times H_{T} \times T}$、$\overline{\mathbf{W}} \in \mathbb{R}^{d^{\prime} \times d \times d}$、$\overline{\mathbf{b}} \in \mathbb{R}^{d^{\prime} \times d}$分别是生成的三个参数的独立参数。$\mathbf{c}_{r}$、$\mathbf{d}_{t} \in \mathbb{R}^{d^{\prime}}$分别表示第$r$个区域和第$t$个时间步的表示。对于$r \in R$，$\mathbf{c}_{r}$是一个自由形式的参数，而对于$t \in T$，$\mathbf{d}_{t}$是根据归一化的一天中的时间特征$\overline{\mathbf{z}}_{t}^{(d)}$和一周中的天数特征$\overline{\mathbf{z}}_{t}^{(w)}$计算得出的。$\mathbf{e}_{1}$、$\mathbf{e}_{2} \in \mathbb{R}^{d^{\prime}}$是它们相应的可学习嵌入。这个参数学习器通过根据特定时间步和区域的特征生成参数，实现了空间和时间的定制。

人话：time emb  经过 mlp 到 time emb , node emb 是nn.Prameter()，然后通过两个emb来生成weight和bias，最后和x emb得到output，这个output最后一个维度是区域数，也就是所谓的分区域吧？

#### 2.1.3 多层次空间模式编码

![20250313162738](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313162738.png)

##### 超图胶囊聚类网络

当前的空间编码器主要聚焦于捕捉局部相邻区域间的关系，却忽略了相距较远区域之间广泛存在的相似性。例如，地理位置上相隔的商业区，仍然可能展现出相似的时空模式。鉴于此，我们的GPT - ST引入了超图胶囊聚类网络，以捕捉全局区域间的相似性。该网络明确学习多个聚类中心作为超边，对不同区域间的相似性进行特征化。为了进一步增强超图结构学习，我们融入了动态路由机制到胶囊网络中。这种机制根据语义相似性，迭代更新超边表示和区域 - 超边连接。因此，它提升了超边的聚类能力，并有助于对每个区域的全局建模依赖性进行建模。

具体而言，我们首先利用之前的嵌入$\boldsymbol{\Gamma}_{r,t}$和挤压函数[30]，在时间步$t$为每个区域$r$获取归一化的区域嵌入$\mathbf{v}_{r,t} \in \mathbb{R}^{d}$。然后，这个嵌入用于计算在第$t$个时间步内，从每个区域$r$到每个聚类中心（超边）$i$的转移信息$\mathbf{v}_{i|r,t} \in \mathbb{R}^{d}$。形式上，这两个变量的计算方式如下：
$$
\mathbf{v}_{r,t} = \text{squash}(\mathbf{V}\boldsymbol{\Gamma}_{r,t} + \mathbf{c}); \quad \mathbf{v}_{i|r,t} = \text{squash}(\mathbf{H}_{i,r,t}^{\prime} \mathbf{v}_{r,t}); \quad \text{squash}(\mathbf{x}) = \frac{\|\mathbf{x}\|^{2}}{1 + \|\mathbf{x}\|^{2}} \frac{\mathbf{x}}{\|\mathbf{x}\|} \tag{5}
$$
其中，$\mathbf{V} \in \mathbb{R}^{d \times d}$和$\mathbf{c} \in \mathbb{R}^{d}$是自由形式的可学习参数。超图连接矩阵$\mathbf{H}_{i,r,t}^{\prime} \in \mathbb{R}^{H_{S} \times R}$记录了$R$个区域和作为聚类中心的$H_{S}$条超边之间的关系。它通过前面提到的定制参数学习器针对第$t$个时间步进行定制，具体为$\mathbf{H}_{i,r,t}^{\prime} = \text{softmax}(\mathbf{d}_{t}^{\prime} \overline{\mathbf{H}}^{\prime})$。这里，$\mathbf{d}_{t}^{\prime}$和$\overline{\mathbf{H}}^{\prime}$是时间特征和超图嵌入。

迭代超图结构学习。随着初始化的区域嵌入$\mathbf{v}_{r,t}$和超图连接嵌入$\mathbf{v}_{i|r,t}$，我们遵循胶囊网络的动态路由机制，以增强超边的聚类效果。第$j$次迭代描述如下：
$$
\mathbf{s}_{i,t}^{j} = \sum_{r = 1}^{R} c_{i,r,t}^{j} \mathbf{v}_{i|r,t}; \quad c_{i,r,t}^{j} = \frac{\exp(b_{i,r,t}^{j})}{\sum_{c} \exp(b_{i,r,t}^{j})}; \quad b_{i,r,t}^{j} = b_{i,r,t}^{j - 1} + \mathbf{v}_{r,t}^{\top} \text{squash}(\mathbf{s}_{i,t}^{j - 1}) \tag{6}
$$
其中，$\mathbf{s}_{i,t} \in \mathbb{R}^{d}$表示迭代的超边嵌入。它是利用迭代的超边 - 区域权重$c_{i,r,t} \in \mathbb{R}$计算得出的。权重$c_{i,r,t}$又由上一次迭代的超边嵌入$\mathbf{s}_{i,t}$计算得到。通过这个迭代过程，关系分数和超边表示会相互调整，以便更好地反映区域和作为空间聚类中心的超边之间的语义相似性。

在动态路由算法的迭代之后，为了同时利用$b_{i,r,t}$和$\mathbf{H}_{i,r,t}^{\prime}$来更好地学习区域 - 超边关系，GPT - ST将两组权重结合起来，并生成最终的嵌入$\overline{\mathbf{s}}_{i,t} \in \mathbb{R}^{d}$。我们首先用$(b_{i,r,t} + \mathbf{H}_{i,r,t}^{\prime})$替换$b_{i,r,t}$，以获得一个新的权重向量$\overline{c}_{i,r,t} \in \mathbb{R}$，然后利用$\overline{c}_{i,r,t} \in \mathbb{R}$来计算最终的嵌入$\overline{\mathbf{s}}_{i,t}$。

（超边似乎是一种可以连接多个聚类中心的边，在代码中设置为16，代码的矩阵运算实在是太复杂了）

##### 跨聚类关系学习

借助聚类后的嵌入$\overline{\mathbf{s}}_{i,t}$，我们提议通过一个高级超图神经网络对聚类间的关系进行建模。具体来说，经过优化的聚类嵌入$\hat{\mathbf{S}} \in \mathbb{R}^{H_{S} \times T \times d}$是通过在$H_{S}$个聚类中心和$H_{M}$个高级超边之间进行消息传递来计算的，如下所示：
$$
\hat{\mathbf{S}} = \text{HyperPropagate}(\tilde{\mathbf{S}}) = \text{squash}(\sigma(\mathbf{H}^{\prime\prime\top} \cdot \sigma(\mathbf{H}^{\prime\prime} \cdot \tilde{\mathbf{S}})) + \tilde{\mathbf{S}}) \tag{7}
$$
其中，$\tilde{\mathbf{S}} \in \mathbb{R}^{H_{S}T \times d}$表示从$i \in H_{S}$且$t \in T$的$\overline{\mathbf{s}}_{i,t}$得到的重塑后的嵌入矩阵。$\mathbf{H}^{\prime\prime} \in \mathbb{R}^{H_{M} \times H_{S}}$表示高级超图结构，它是通过前面提到的个性化参数学习器获得的。这种参数定制将所有$t \in T$的时间特征$\overline{\mathbf{z}}_{t}^{(d)}$、$\overline{\mathbf{z}}_{t}^{(w)}$聚合作为输入，并按照公式(4)生成参数。

在优化了$i \in H_{S}$，$t \in T$时的聚类表示$\hat{\mathbf{s}}_{i,t} \in \mathbb{R}^{d}$之后，我们将聚类后的嵌入通过低级超图结构传播回区域嵌入，如下所示：
$$
\boldsymbol{\Psi}_{r,t} = \sigma(\sum_{i = 1}^{H_{S}} c_{i,r,t} \cdot \hat{\mathbf{s}}_{i,t} \mathbf{W}_{r}^{\prime\prime} + \mathbf{b}_{r}^{\prime\prime} + \boldsymbol{\Gamma}_{r,t}) \tag{8}
$$
其中，$\boldsymbol{\Psi}_{r,t} \in \mathbb{R}^{d}$表示第$t$个时间步中第$r$个区域的新区域嵌入。$c_{i,r,t} \in \mathbb{R}$表示低级超图胶囊网络的权重。$\mathbf{W}_{r}^{\prime\prime} \in \mathbb{R}^{d \times d}$和$\mathbf{b}_{r}^{\prime\prime} \in \mathbb{R}^{d}$表示由定制参数学习器生成的区域特定变换和偏差参数。

#### 2.1.4 Cluster-aware Masking Mechanism

![20250313162259](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313162259.png)

受语义引导的 MAE[21]启发，我们为 GPT-ST 设计了一种聚类感知掩码机制，以增强其簇内和簇间关系学习。自适应掩码策略结合了前面学到的聚类信息 $\bar{c}_{i, r, t}$，以开发一种由易到难的掩码过程。具体来说，在训练开始时，我们为每个簇随机掩码一部分区域，在这种情况下，被掩码的值可以通过参考具有相似 ST 模式的簇内区域轻松预测。随后，我们逐渐增加某些类别的掩码比例，通过减少它们的相关信息来增加这些簇的预测难度。最后，我们完全掩码一些簇的信号，促进预训练模型的跨簇知识转移能力。这个自适应掩码过程如图 4 所示。然而，直接利用学到的聚类信息 $\bar{c}_{i, r, t}$ 来生成聚类感知掩码 M 是不可行的。这是因为聚类信息是由我们的 GPTST 网络的更深层计算的，而生成的掩码需要作为网络的输入（$(M \odot X)$）。为了解决这个问题，我们使用具有定制参数的两层 MLP 网络来预测 $\bar{c}_{i, r, t}$ 的学习结果。具体来说，这个 MLP 网络中的变换和偏差向量被由定制参数学习器生成的时间动态和节点特定参数所取代（公式 4）。随后，使用线性层和 softmax(·)函数来获得对 $i \in H_{S}$、$r \in R$、$t \in T$ 的预测。为了优化 $q_{i, r, t}$ 的分布，我们使用具有真实值 $\bar{c}_{i, r, t}$ 的 KL 散度损失函数 $L_{k l}$。需要注意的是，在这一步中阻止了 $\bar{c}_{i, r, t}$ 的反向传播。根据 $q_{i, r, t}$，取最大概率的类别作为分类结果。更多细节见 A.2 中的算法 1。

人话：time emb  经过 mlp 到 time emb , node emb 是nn.Prameter()，然后通过两个emb来生成weight和bias，最后和x emb得到output，output又做了sofmax来获得预测，因为output最后一个维度是聚类数，所以这算是一个显示的聚类？

人话：关于自适应掩码，总而言之就是根据上述的output得到一个自适应的选择，然后经过某种处理和random掩码乘在一起，这不就是课程学习吗。

### 2.2 模型结构

![20250313162807](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313162807.png)

![20250313153115](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313153115.png)

从未见过有如此复杂的模型

## 三、实验验证与结果分析

### 3.1 消融实验

![20250313162841](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313162841.png)

论文中的消融实验主要用于探究GPT-ST中各主要组件的作用，通过对多个消融变体重新预训练，并使用新预训练的模型评估下游方法的性能，以分析不同组件对模型的影响，具体内容如下：

1. **基本组件的影响**
    - **-P**：移除定制参数学习器，结果显示性能显著下降，表明生成个性化时空参数对捕获复杂时空相关性和提升预测有积极作用。
    - **-C**：禁用超图胶囊聚类网络，取消聚类操作，空间超图神经网络直接在细粒度区域上工作，性能下降明显，因为聚类结果在其他组件（如跨簇依赖和聚类感知自适应掩码）中也很重要。
    - **-T**：移除高级空间超图，不探索跨簇关系学习，性能同样下降，说明建模簇内和簇间的时空依赖能有效捕获复杂时空相关性，对预测有益。
2. **掩码机制的影响**
    - **Ran0.25和Ran0.75**：分别用掩码率为0.25（与原策略相同）和0.75（MAE和STEP中使用的掩码率）的随机掩码策略替代自适应掩码策略，结果表明自适应掩码策略优于随机掩码策略，因其有效促进了GPT-ST对簇内和簇间关系的学习，生成了高质量的表示。
    - **GMAE和AdaMAE**：将该方法与使用GraphMAE和AdaMAE掩码策略的变体进行比较，这两个变体性能均不如原方法，凸显了在掩码策略中考虑时空模式的重要性，进一步证实了利用聚类信息的自适应掩码方法的优越性。
3. **预训练策略的影响**：将掩码重建预训练方法与其他预训练方法（局部-全局信息最大化和对比预训练，以DGI和GraphCL为基线）进行比较。结果显示，虽然后两者相比无预训练模型有性能提升，但GPT-ST的掩码重建任务由于与下游回归任务相关性更高，能更有效地学习时空表示，且自适应掩码策略通过增加预训练任务难度，在促进模型学习稳健时空表示方面也发挥了关键作用，使得该方法性能提升最为显著。

### 3.2 聚类效果调查

![20250313163239](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250313163239.png)

1. **可视化嵌入分析**：运用T-SNE算法将超图胶囊聚类网络产生的高维嵌入映射到二维向量，用不同颜色代表不同类别（超参数$H_{S}$定义为10类 ），根据区域属于不同类别的概率确定区域聚类。观察可视化结果，发现同一类别的区域在有限空间内紧密聚集，为超图胶囊聚类网络强大的聚类能力提供了实证依据。
2. **区域关系案例研究**：在另一个对 METR-LA 数据集进行的案例研究中，我们研究了超图胶囊聚类网络得出的簇内区域关系，以及跨簇超图网络获得的跨类依赖关系。结果如图 7 所示，表明同一类别中的前三个区域，如图 7 (a) 和 7 (b) 所示，呈现出相似的交通模式，表明具有共同的功能。例如，商业区附近的区域（7 (a)）在晚上出现高峰，而住宅区附近的区域（7 (b)）则保持相对稳定。这些观察结果与现实世界的情况一致。此外，关注图 7 (c)，我们分析了两个类别中的前两个区域，这些区域在特定时间段内交通模式发生了变化，同时在跨类过渡中具有相似的超边权重。结果表明，经历模式转变的区域呈现出不同的交通模式，同时在短驾驶距离内保持紧密的相互联系。这些发现进一步证明了跨簇过渡学习能够捕捉区域之间的语义级关系，反映现实世界的交通场景。这些优势有助于在我们的 GPT-ST 框架中生成高质量的表示，从而提高下游任务的性能。
