# 机器学习(?)术语表

## 这里已经包含了的名词

现在只有很少一部分名词的解释。当然，以后的一段时间这里的名词会变得越来越多。

协变量协变量偏移 , 量纲 , 无量纲化 , 数据集 , 输入 , 输出 , 样本 , 特征 , 特征图 , 批量标准化 , 标准化 , 归一化 , 中心化 , 标准化、归一化、中心化的区别 , 过拟合 , 欠拟合 , 特征缩放 , 损失函数 , 准确率 , 随机梯度下降 , 有监督学习 , 无监督学习 , 半监督学习 , 向前传播 , 向后传播 

查找一个词条的办法是按下键盘上的`Ctrl+F`，然后键入关键字。

## 拼音首字母索引：

---

## 英文首字母索引：

### 字母 A

- 准确率（accuracy）「[查看来源](https://developers.google.com/machine-learning/glossary/#%E5%87%86%E7%A1%AE%E7%8E%87-accuracy)」

  [**分类模型**](https://developers.google.com/machine-learning/glossary/#classification_model)的正确预测所占的比例。在[**多类别分类**](https://developers.google.com/machine-learning/glossary/#multi-class)中，准确率的定义如下：
$$
  \text{准确率} = \frac{\text{正确的预测数}} {\text{样本总数}}
$$

在[**二元分类**](https://developers.google.com/machine-learning/glossary/#binary_classification)中，准确率的定义如下：

$$
  \text{准确率} = \frac{\text{正例数} + \text{负例数}} {\text{样本总数}}
$$

请参阅[**正例**](https://developers.google.com/machine-learning/glossary/#TP)和[**负例**](https://developers.google.com/machine-learning/glossary/#TN)。

- A/B testing （A/B 测试）「[查看来源](https://developers.google.com/machine-learning/glossary/#ab-%E6%B5%8B%E8%AF%95-ab-testing)」

  一种统计方法，用于将两种或多种技术进行比较，通常是将当前采用的技术与新技术进行比较。A/B 测试不仅旨在确定哪种技术的效果更好，而且还有助于了解相应差异是否具有显著的统计意义。A/B 测试通常是采用一种衡量方式对两种技术进行比较，但也适用于任意有限数量的技术和衡量方式。

---

### 字母 B

- 向后传播（反向传播，backwarding，back propagation）

- 批量标准化（BN，batch normalization）

  BN 是由 Google 于 2015 年提出，这是一个深度神经网络训练的技巧，它不仅可以加快了模型的收敛速度，而且更重要的是在一定程度缓解了深层网络中“梯度弥散”的问题，从而使得训练深层网络模型更加容易和稳定。所以目前 BN 已经成为几乎所有卷积神经网络的标配技巧了。

---

### 字母 C

- 分类模型（classification model）「[查看来源](https://developers.google.com/machine-learning/glossary/#%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B-classification-model)」

- 协变量（covariate）

  在机器学习和深度学习方法中，协变量指的是与[输入](//todo)无关的其他变量。在计算机视觉中，输入一般指的是图像或特征图（[feature map](//todo)），而协变量可以指权重 w 等。举个例子：

  $$
  降雨总量S(t) = KTt + e
  $$

  其中，t 是自变量时间，降雨量（t）是因变量，而温度（T）则是协变量，K 为一个常数。

  通常情况下，在实验的设计中，协变量是一个独立变量（解释变量），不为实验者所操纵，但仍影响实验结果。

- 协变量偏移（covariate shift）

---

### 字母 D

- 数据集（dataset）

- 解码器（decoder）

- 无量纲化（dimensionless，nondimensionalize）

  当在某种运算中，两种不同量纲的变量对运算结果起着同等的作用，或它们对结果的重要性一样，它们往往可以当作同类的量处理，或同时进行[归一化](//todo)处理

---

### 字母 E

- 编码器（encoder）

---

### 字母 F

- **特征**（feature）

- **特征图**（ feature map）

- **向前传播**（正向传播，forwarding，forward propagation）

- **量纲**（因次，fundamental unit）

  物理学中，不同的物理量有着不同的单位，然而这些单位之间都有相互的联系。实际上，恰当地规定一些基本的单位（称为基本单位），可以使任何其他的单位（称为导出单位）都表达为这些单位的乘积，将其统一以便于研究各个物理量之间的关系。如在国际单位制中，功的单位焦耳（$J$），可以表示为“千克平方米每平方秒”（$kg \cdot m^2 / s^2$）。

  然而，仅仅用单位来表示会面临一些问题：

  1. 在不同的单位制下，各个物理量用单位来表示也会不同，以至于起不到预期的“统一各单位”的效果。如英里每小时（$mph$）与米每秒（$m/s$）乍看之下无甚联系，然而它们却都是表示速度的单位。虽然说经过转换可以将各个基本单位也统一，然而这样终究不够直观，需记忆也不甚方便，而且选择哪一个单位作为统一单位似乎都不甚公平。
  2. 把一个既有的单位表达为拆分了的基本单位的形式实际上没有任何意义，功的单位无论如何都不是“千克二次方米每二次方秒”，因为实际上这个单位根本不存在，它只是与“焦耳”恰好相等而已。况且，这样做也会导致一些拆分后相同但实质不同的单位被混淆，如力矩的单位牛米（$N \cdot m$）被拆分后也是（$kg \cdot m^2 /s^2$），然而它与功显然是完全不同的。

  因此量纲被作为表达导出单位组成的专有方式引入物理学中。

- **特征缩放**（feature scaling）

  正在施工

- **前馈神经网络** ( feedforward neural network , FNN )

  前馈神经网络（[英文](https://zh.wikipedia.org/wiki/英文)：Feedforward Neural Network），为[人工智能](https://zh.wikipedia.org/wiki/人工智能)領域中，最早发明的简单[人工神经网络](https://zh.wikipedia.org/wiki/人工神经网络)类型。在它内部，参数从输入层向输出层单向传播。有异于[循环神经网络](https://zh.wikipedia.org/wiki/循环神经网络)和[递归神经网络](https://zh.wikipedia.org/wiki/递归神经网络)，它的内部不会构成[有向环](https://zh.wikipedia.org/wiki/環_(圖論))。FNN由一个输入层、一个（浅层网络）或多个（深层网络，因此叫作深度学习）隐藏层，和一个输出层构成。每个层（除输出层以外）与下一层连接。这种连接是 FNN 架构的关键，具有两个主要特征：加权平均值和激活函数。

  一般情况下，前馈神经网络被分为[单层感知机](https://zh.wikipedia.org/wiki/感知机)和[多层感知机](https://zh.wikipedia.org/wiki/多层感知机)。

---

### 字母 G

---

### 字母 H

---

### 字母 I

- 输入（输入数据，input，input data，input dataset）

---

### 字母 J

---

### 字母 K

---

### 字母 L

- 标签（label）

- 损失函数（loss function）

---

### 字母 M

- **多层感知机**（人工神经网络，MLP，multilayer perceptron，ANN，artificial neural network）

  多层感知器（Multilayer Perceptron,缩写MLP）是一种[前向结构](https://zh.wikipedia.org/wiki/前馈神经网络)的[人工神经网络](https://zh.wikipedia.org/wiki/人工神经网络)，映射一组输入向量到一组输出向量。MLP可以被看作是一个有向图，由多个的节点层所组成，每一层都全连接到下一层。除了输入节点，每个节点都是一个带有非线性激活函数的神经元（或称处理单元）。一种被称为[反向传播算法](https://zh.wikipedia.org/wiki/反向传播算法)的[监督学习](https://zh.wikipedia.org/wiki/监督学习)方法常被用来训练MLP。多层感知器遵循人类神经系统原理，学习并进行数据预测。它首先学习，然后使用权重存储数据，并使用算法来调整权重并减少训练过程中的偏差，即实际值和预测值之间的误差。主要优势在于其快速解决复杂问题的能力。多层感知的基本结构由三层组成：第一输入层，中间隐藏层和最后输出层，输入元素和权重的乘积被馈给具有神经元偏差的求和结点,主要优势在于其快速解决复杂问题的能力。MLP是[感知器](https://zh.wikipedia.org/wiki/感知器)的推广，克服了感知器不能对[线性不可分](https://zh.wikipedia.org/w/index.php?title=线性不可分&action=edit&redlink=1)数据进行识别的弱点。

---

### 字母 N

- 样本归一化（归一化，normalization）

  通常来说，样本标准化是指特征工程中的[特征缩放](//todo)过程。

  在数值上，归一化一般将把数据变成$(0,1)$或$(-1,1)$之间的小数。

  归一化/标准化实质是一种线性变换，线性变换有很多良好的性质，这些性质决定了对数据改变后不会造成“失效”，反而能提高数据的表现，这些性质是归一化/标准化的前提。比如有一个很重要的性质：线性变换不会改变原始数据的数值排序。

  在使用梯度下降的方法求解最优化问题时， 归一化/标准化后可以加快梯度下降的求解速度，即提升模型的收敛速度。

  简单的线性数值归一化运算常见的表示形式有：

  1. Min-Max Normalization： $x' = (x - min(x)) / (max(x) - min(x))$
  2. 平均归一化：$x' = (x - μ) / (max(x) - min(x))$

  简单的线性标准化可能存在的缺陷是当有新数据加入时，可能导致$max(x)$和$min(x)$的变化，需要重新定义。

  非线性的归一化可能的表示形式有：

  1. 对数函数转换：$x' = \log(x)$
  2. 反余切函数转换：$x' = \arctan(x) \cdot 2 / π$

  非线性的标准化方法经常用在数据分化比较大的场景，即有些数值很大，有些很小。通过一些数学函数（包括 log、指数，正切等），将原始值进行映射。根据数据分布的情况，可能会决定使用不同的非线性函数的曲线。并没有完全通用的标准化方法。

- 无量纲化（nondimensionalize，dimensionless）

  当在某种运算中，两种不同量纲的变量对运算结果起着同等的作用，或它们对结果的重要性一样，它们往往可以当作同类的量处理，或同时进行[归一化](//todo)处理

---

### 字母 O

- 输出（输出数据，output，output dataset）
- 过拟合（overfitting）
- one-hot编码（onehot，one hot encoding）
- 

---

### 字母 P

- **感知机**

---

### 字母 Q

---

### 字母 R

- 随机梯度下降（random / stochastic gradient descent）

---

### 字母 S

- 样本（sample）

- 标准化（样本标准化，standardization）

  在机器学习和深度学习中，我们可能要处理不同种类的资料，例如，音讯和图片上的像素值，这些资料可能是高维度的，资料标准化后会使每个特征中的数值平均变为 0(将每个特征的值都减掉原始资料中该特征的平均)、标准差变为 1，这个方法被广泛的使用在许多机器学习算法中(例如：支持向量机、逻辑回归和类神经网络)。

- 随机梯度下降（random / stochastic gradient descent）

- 有监督学习（监督学习，监督式学习，supervised learning）

  一句话讲完：通过已有的一部分输入数据与输出数据之间的对应关系，生成一个函数，将输入映射到合适的输出，例如分类。

  多说一点：监督式学习（Supervised learning），是一个机器学习中的方法，可以由训练资料中学到或建立一个模式（ learning model），并依此模式推测新的实例。训练资料是由输入物件（通常是向量）和预期输出所组成。函数的输出可以是一个连续的值（称为回归分析），或是预测一个分类标签（称作分类）。

---

### 字母 T

---

### 字母 U

- 欠拟合（underfitting）

- 半监督学习（unsupervised learning）

  综合利用有类标的数据和没有类标的数据，来生成合适的分类函数。

- 无监督学习（非监督学习，unsupervised learning）

  直接对输入数据集进行建模，例如聚类。

---

### 字母 V

---

### 字母 W

---

### 字母 X

---

### 字母 Y

---

### 字母 Z

- 中心化（零均值化，zero centered）

  中心化后要求样本平均值为 0，对标准差无要求

---

# Other tips

- 标准化、归一化、中心化的区别

1. 标准化和中心化的区别：标准化是原始分数减去平均数然后除以标准差，中心化是原始分数减去平均数。 所以一般流程为先中心化再标准化。
2. 归一化和标准化的区别：归一化是将样本的特征值转换到同一量纲下把数据映射到[0,1]或者[-1, 1]区间内，仅由变量的极值决定，因区间放缩法是归一化的一种。标准化是依照特征矩阵的列处理数据，其通过求 z-score 的方法，转换为标准正态分布，和整体样本分布相关，每个样本点都能对标准化产生影响。它们的相同点在于都能取消由于量纲不同引起的误差；都是一种线性变换，都是对向量 X 按照比例压缩再进行平移。



---



# 前方正在施工

- 上采样（up sampling）
- 下采样（降采样，down sampling,sub sampling）
- 梯度弥散
- 梯度爆炸
- 梯度消失
- 特征缩放
- 循环神经网络（RNN，recursive neural network）
- 长短期记忆（LSTM，long short term memory）
- 前馈神经网络（feedforward neural network）