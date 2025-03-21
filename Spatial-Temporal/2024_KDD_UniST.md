# UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction

>领域：时空数据预测  
>发表在：KDD 2024  
>模型名字：***Uni***versal ***S***patio-***T***emporal Prediction  
>文章链接：[UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction](https://arxiv.org/abs/2402.11838)  
>代码仓库：[https://github.com/tsinghua-fib-lab/unist](https://github.com/tsinghua-fib-lab/unist)  
![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307192417.png)

## 一、研究背景与问题提出

### 1. 1 研究现状

预训练基础模型在自然语言处理（NLP）中取得了显著成功，尤其在少样本和零样本设置下表现出色 。然而，在城市时空预测领域尚未实现类似的突破。在本文中，我们的目标是为通用城市时空预测建立一个基础模型。

一个通用的时空模型必须具备两个基本能力。首先，它必须能够利用来自不同城市场景的丰富多样的数据进行训练。基础模型的训练应确保获取充足而丰富的信息。其次，它应该在不同的时空场景中表现出强大的泛化能力。特别是在训练数据有限或没有训练数据的场景中，该模型仍然可以良好地工作，而不会出现明显的性能下降 。

### 1.2 现存问题

然而，实现上述能力会遇到时空数据特有的重大挑战，这阻碍了为语言和视觉领域开发的当前基础模型的直接应用。

1. 第一个挑战来自时空数据集固有的不同格式。与具有自然统一顺序结构的语言或遵循标准化维度的图像和视频不同，***从不同来源收集的时空数据表现出高度多样化的特征***。这些特征包括可变维度、时间跨度和空间。差异显著的覆盖范围，给标准化其结构带来了困难。

2. 第二个挑战来自多个场景中数据分布的高度变化。***面对高度不同的时空模式，模型可能难以适应这些差异***。与语言不同，语言受益于共享词汇表，不同领域和城市的各种场景通常在完全不同的时空尺度上运行，缺乏有效训练和泛化的共同要素。

与具有简单顺序结构的时间序列不同，时空数据具有更复杂的性质，在空间和时间维度上都存在相互交织的依赖关系。虽然探索大型语言模型的整合很有前景，但重要的是要认识到时空数据并非由语言内在生成。因此，开发专门在纯时空数据上训练的基础模型也是一个重要方向。  
![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307234335.png)

## 二、问题剖析与解决策略

### 2.1 解决方法

UniST 通过由数据、架构、预训练和提示学习这四个关键组件驱动的整体设计来实现上述能力。

首先，我们利用来自不同领域和城市的大量数据，充分挖掘时空场景中固有的丰富多样性。其次，我们设计了***时空补丁技术将多样的数据统一为序列格式***，便于利用强大的 Transformer 架构。

第三，受大型语言模型和视觉模型的启发，UniST 采用了广泛应用的生成式预训练策略 —— 掩码标记建模（MTM）。我们通过采用***多种掩码策略来全面处理多视角相关性，进一步增强了模型捕捉复杂时空关系的能力***。

此外，基于时空建模领域已有的知识，我们设计了一种创新的***提示学习方法***。精心设计的提示网络能够识别潜在的、共享的时空模式，并动态调整以生成有用的提示。

#### 2.1.1 时空补丁技术

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307234608.png)

#### 2.1.2 掩码Token建模

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307234624.png)

#### 2.1.3 提示学习

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307234714.png)  

1. 空间邻近性（Spatial closeness）附近的单元（这里的 “单元” 可以是地理区域、交通站点、建筑物等在空间上的一个个实体或区域）之间可能会相互影响。
2. 空间层次结构（Spatial hierarchy）空间的层次化组织会对时空动态产生影响，这就需要对城市结构有一个多层次的认知。城市结构通常具有一定的层次，比如从宏观层面的城市分区（如商业区、住宅区、工业区等），到中观层面的街区，再到微观层面的具体建筑或设施。不同层次之间存在着相互作用和影响，并且在时空动态上也有所体现。
3. 时间邻近性（Temporal closeness）近期的动态变化会影响未来的结果，这表明存在一种邻近性依赖关系。
4. 时间周期性（Temporal period）每日或每周的模式会表现出相似性，呈现出一定的周期性。

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250308000631.png)  
为了简化操作，我们给出了一些简单直接的实现方法，这些方法展示在图2中的四个网络里，也就是 $NET _{t c}$、$NET _{t}$ 、$NET _{s c}$ 和 $NET _{s h}$ 这几个网络。

1. 在空间维度上，我们首先运用一种注意力机制，把时间维度融合到一个被称作 $E_{s}$ 的表示当中。接着，为了捕捉近距离范围内的空间依赖关系，我们采用了一个二维卷积神经网络 $CNN$ ，也就是卷积核大小为3的 $NET _{S C}$ 网络。为了捕捉空间层次结构信息，我们使用了具有更大卷积核的CNN，也就是 $NET _{s h}$ 网络。这些更大的卷积核能够在更大的尺度上感知空间信息，这有助于构建出一个层次化的视角。

2. 在时间维度方面，我们采用了一个注意力网络，即 $NET _{tc }$ 网络，来聚合前面的 $M$ 个时间步长的数据（用 $X_{c}$ 来表示）。对于时间周期，我们从过去的 $N$ 天中选取相应的时间点，将其记为 $x_{p}$ 。随后，我们采用另一个注意力网络，也就是 $NET _{t p}$ 网络，来聚合这个周期性的序列，这样做能够捕捉到长期的时间模式。

整个过程可以用以下公式来表达：
$$E_{s c}=Conv_{2 D}[3]\left(X_{s}\right),$$
这表明 $E_{s c}$ 是通过对 $X_{s}$ 进行卷积核大小为3的二维卷积运算得到的。

$$E_{s h}=\left\{Conv_{2 D}\left[2^{i}+1\right]\left(X_{s}\right)\right\}, i \in\{2,3,4\},$$
这意味着 $E_{s h}$ 是由对 $X_{s}$ 进行不同卷积核大小 $2^{i}+1$，其中 $i$ 取值为 2、3、4 的二维卷积运算的结果组成的集合。

$$E_{t c}=Attention\left(X_{c}\right) ,$$
表示 $E_{t c}$ 是对 $X_{c}$ 进行注意力运算得到的结果。

$$E_{t p}=ATTENTION\left(X_{p}\right) .$$

表示 $E_{t p}$ 是对 $X_{p}$ 进行注意力运算，从而聚合周期性序列得到的结果。

### 2.2 模型结构

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307234737.png)

## 三、实验验证与结果分析

### 3.1 Performance Study

#### 3.1.1 短期预测：6->6

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307234959.png)

#### 3.1.2 长期预测：64->64

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307235101.png)

#### 3.1.3 Few-shot Learning

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307235259.png)

#### 3.1.4 Zero-shot Learning

见上图

### 3.2 消融实验

![1](https://picgo-for-paper-reading.oss-cn-beijing.aliyuncs.com/img/20250307235542.png)  

用 “s” 代表空间邻近性和层次结构，“p” 代表时间周期性，“c” 代表时间邻近性。
