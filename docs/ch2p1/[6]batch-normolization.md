# 标准化处理

批量标准化（Batch Normalization，有时也称批量规范化、批量归一化、批量高斯化）方法，能大幅加速模型训练，同时保持预测准确率不降，因而被一些优秀模型采纳为标准模型层。

首先说明结论：**批量标准化方法的效果很好，很多框架已经把BN作为一种基本的网络层放在框架里给大家用了**。如果你现阶段并不打算深究其原理，可以不再往下看了。

> 推荐你学完基础的概率论与数理统计知识后再回来继续阅读。

批量标准化是在一篇称为[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)的论文中提出的。这篇论文指出，由于在深度神经网络模型中人们常设置较低的学习率，并谨慎地进行参数初始化，很多网络还具有饱和非线性的激活函数，这就让训练速度变得更慢更困难。这篇论文将此现象称为内部协变量平移。

训练一个在训练集和真实情况下效果都不错的深度神经网络非常困难，因为在训练过程中，随着先前各层中参数发生变化，每一层输入的分布也会发生变化。而数据分布对训练会产生影响。

**所以，神经网络中的标准化处理的基本目的是为了解决协变量偏移（**Covariate Shift**）**。如果你暂时不清楚协变量的意思，可以暂时理解为神经网络的输入（记为$X$）就是协变量，有时也称为解释变量（ explanatory variable）。

以下是这篇论文的摘要：

> Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training, as the parameters of the previous layers change. This slows down the training by requiring lower learning rates and careful parameter initialization, and makes it notoriously hard to train models with saturating nonlinearities. We refer to this phenomenon as internal covariate shift, and address the problem by normalizing layer inputs. Our method draws its strength from making normalization a part of the model architecture and performing the normalization for each training mini-batch. Batch Normalization allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer, in some cases eliminating the need for Dropout. Applied to a state-of-the-art image classification model, Batch Normalization achieves the same accuracy with 14 times fewer training steps, and beats the original model by a significant margin. Using an ensemble of batch-normalized networks, we improve upon the best published result on ImageNet classification: reaching 4.9% top-5 validation error (and 4.8% test error), exceeding the accuracy of human raters.

这篇论文包含了大量重要的推理流程和信息，如果有兴趣，请阅读[原论文](https://arxiv.org/abs/1502.03167)。这篇笔记只是一个粗浅的介绍。

## 协变量偏移

如果训练数据和待预测数据的条件概率相同，而边缘概率不同:
$$
P_{train}(y|x) = P_{te}(y|x)\\
P_{train}(x) \ne P_{te}(x)\\
$$
其中，${}_{train}$角标表示训练数据，即$P_{train}(y|x)$中的$x$和$y$分别代表训练时的输入及真实值；${}_{te}$角标表示待预测的数据，即$P_{te}(y|x)$中的$x$和$y$分别代表待预测值的输入及真实值。

这种训练数据和待预测数据的条件概率相同，而边缘概率不同情况就被称为协变量偏移（Covariate Shift）。

我们回顾概率论知识，在条件概论中，有：
$$
P(x,y) = P(y|x)P(x)
$$
所以在协变量偏移的条件下，就存在：
$$
P_{train}(x,y) \ne P_{te}(x,y)
$$
其证明过程是：

consider the following statements:
$$
\begin{aligned}
P(x,y) = P(y|x)P(x)\qquad(1)\\
P_{train}(y|x) = P_{te}(y|x)\qquad(2)\\
P_{train}(x) \ne P_{te}(x)\qquad(3)\\
P_{train}(x,y) \ne P_{te}(x,y)\qquad(4)
\end{aligned}
$$
we have:
$$
1,2,3\rightarrow 4
$$
也就是说，**此时训练数据和待预测数据的联合概率也不同的**，这会**导致训练出来的模型预测失准**。

## Batch Normalization（BN）

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)的作者希望通过减少层间协变量偏移，改善模型训练效率，以此为初衷，提出了`Batch Normalization`方法。

原论文中 Normalization via Mini-Batch Statistics 的部分证明了分Batch对数据进行标准化的有效性，这里就不赘述其证明，只大致叙述其推理过程。

模型训练前向传播时，设每个容量为m的mini-batch是样本集合X，对其中每个单一的$x$, 采用四步进行Batch Normalization：

- （1）mini-batch mean
- （2）mini-batch variance
- （3）normalize
- （4）scale and shift

原论文给出了以下公式：
$$
\begin{aligned}
{\mu}_{batch} = \frac{1}{m}\sum_{i=1}^m x_i \qquad(1)\\
\sigma_{batch}^2 = \frac{1}{m}\sum_{i-1}^m (x_i - \mu_{batch})^2 \qquad(2)\\
\hat(x)_{i} = \frac{x_i - \mu_{batch}}{\sqrt{\sigma_{batch}^2 + \epsilon}} \qquad(3)\\
y_i = \gamma\hat{x}^i + \beta \equiv {BN}_{\gamma,\beta}(x_i) \qquad(4)\\
\end{aligned}
$$
其中，式(3)分母的$\epsilon$（$1e-8$）是平滑项(smoothing term), 用于在方差极小的情况下，避免除0( avoids division by zero) 。

观察前三步，把输入数据规范化为**均值为0**，**方差近似1**的标准化变量，再通过式（4）的线性变换，做伸缩(scaling)和平移（shifting），通过γ和β参数训练，学习合适的伸缩与平移幅度，恢复模型表达，得到mini-batch的Batch Norm输出；训练完成后，学习得 γ 和 β 参数。

Batch Normalization通过一个稍微复杂一些的数学原理（可以查看原论文，这里就不赘述了）把输入数据规范化为**近似标准正态分布**, 再通过学习得到的参数，对数据做缩放和平移，恢复特征表达；输出的数据，仍然近似的服从。



## 神经网络中的BN

原论文中分开讨论了对数据预处理时的BN、线性层的BN以及卷积层的BN。BN的大致过程已经给出了，这部分在原论文中主要是有效性的推理。如果有兴趣可以查看原论文。（或者也许我下次有空整理了会整理在这里）。

卷积层输出特征张量的通道数，由卷积核的个数决定，每个输出通道（深度）上的特征张量（feature map），其不同区域使用同一组γ、β参数做Batch Norm处理。

## 小结

从结果看，深层网络，普遍使用BN取得更好的训练效果：

- 缓解了**梯度传递问题**，使模型适应更大的学习率，加速了训练；
- 改善了**饱和非线性模型**不易训练的问题；
- 还起到了**正则化**的作用。

原论文中的说法是：

- Batch Normalization enables higher learning rates
- Batch Normalization regularizes the model