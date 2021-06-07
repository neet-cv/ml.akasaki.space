# Transformer: Attention is all you need

> 论文名称：[Attention Is All you Need](https://arxiv.org/pdf/1706.03762.pdf)
>
> 作者：Ashish Vaswani，Noam Shazeer，Niki Parmar，Jakob Uszkoreit，Llion Jones，Aidan N. Gomez，Łukasz Kaiser，Illia Polosukhin
>
> code：https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py

### 这篇笔记的写作者是[AsTheStarsFall](https://github.com/asthestarsfalll)。

## 前言

基于RNN或CNN的Encoder-Decoder模型在NLP领域占据大壁江山，然而她们也并非是完美无缺的：

- LSTM，GRU等RNN模型受限于固有的循环顺序结构，无法实现**并行计算**，在序列较长时，计算效率尤其低下，虽然最近的工作如[因子分解技巧](http://arxiv.org/abs/1703.10722)[^1]，[条件计算](https://arxiv.org/abs/1701.06538)[^2]在一定程度上提高了计算效率和性能，但是顺序计算的限制依然存在；
- Extended Neural GPU[^3],[ByteNet](https://arxiv.org/abs/1610.10099)[^4],和[ConvS2S](https://arxiv.org/abs/1705.03122)[^5] 等CNN模型虽然可以进行并行计算，但是学习任意两个位置的信号的长距离关系依旧比较困难，其计算复杂度随距离线性或对数增长。

而谷歌选择抛弃了主流模型固有的结构，提出了**完全**基于注意力机制的Transformer，拥有其他模型无法比拟的优势：

- Transformer可以高效的并行训练，因此速度十分快，在8个GPU上训练了3.5天；
- 对于长距离关系的学习，Transformer将时间复杂度降低到了常数，并且使用多头注意力来抵消位置信息的平均加权造成的有效分辨率降低
- Transform是一种自编码（Auto-Encoding）模型，能够同时预测上下文

## 整体结构

Transfromer的整体结构是一个Encoder-Decoder，自编码模型主要应用于语意理解，对于生成任务还是自回归模型更有优势

![image-20210605151335569](https://gitee.com/Thedeadleaf/images/raw/master/image-20210605151335569.png)

我们可以将其分为四个部分：输入，编码块，解码块与输出

![](https://gitee.com/Thedeadleaf/images/raw/master/photo_2021-06-05_15-27-55.jpg)

接下来让我们按照顺序来了解整个结构，希望在阅读下文前你可以仔细观察这幅图，阅读时也请参考该图

### 输入

使用Word2Vec等方法进行Word Embedding，论文中嵌入维度$d_{model}=512$

得到词嵌入向量后需要除以$\sqrt{d_{model}}$来进行缩放

同时会将上一层的输出加入进来，网络的第一层则会直接使用Inputs充当“上一层”

在输入之后会进行**位置编码**，使得Transformer拥有捕捉序列顺序的能力

### Encoder-Decoder

整体结构如图

![image-20210606010226377](https://gitee.com/Thedeadleaf/images/raw/master/image-20210606010226377.png)

Encoder-Decoder的内部结构如下图：

![image-20210606010346416](https://gitee.com/Thedeadleaf/images/raw/master/image-20210606010346416.png)

- **Encoder**：编码块是由6个完全相同的layer组成的，每个layer有两个子层。

  第一层包括一个$Multi-Head Self-Attention$、$Layer-Normalization$和残差连接，$FFN(x)=max(0,xW_1+b_1)W_2+b_2$,中间层的维度为2048

  第二层包括一个二层的全连接前馈层（中间使用ReLU）、$Layer-Normalization$和残差连接，

- **Decoder**：解码块同样由6个完全相同的layer组成，每个子层同样有残差连接和$Layer-Normalization$

  额外添加了第三个子层——$Masked-Multi-Head-Attention$，这是针对于上一层输出的，将在下文详细解读

  此外，还修改了子注意力子层（如上图，由原来的Self-Attention变Encoder-Decoder Attention）

**Layer Normalization**：NLP任务中主要使用$Layer-Norm$而不是$Batch-Norm$，因为在批次上进行归一化会混乱不同语句之间的信息，我们需要在每个语句之中进行归一化。

### 输出

对解码器的输出使用普通的线性变化与$Softmax$，作为下一层的输入

## 注意力机制

### Self-Attention

具体内容可参考我的另一篇博客——[注意力机制](https://asthestarsfalll.icu/2021/05/12/%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6/index.html)

### 缩放点积注意力

缩放点积注意力，图式如下：

![image-20210606005714922](https://gitee.com/Thedeadleaf/images/raw/master/image-20210606005714922.png)

其公式为
$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
缩放点积指的是其中的打分函数
$$
\frac{QK^T}{\sqrt d}
$$
常见的注意力模型有加性模型和点积模型，点积模型相较于加性模型效率更高，但是当输入向量维度过高，点积模型通常有较大的方差，从而导致softmax函数**梯度很小**，而缩放点积模型可以很好地解决这个问题。

### Multi-Head Attention

多头注意力，图式如下

![image-20210606012114964](https://gitee.com/Thedeadleaf/images/raw/master/image-20210606012114964.png)

相比于使用$d_{model}$维数（此处为512维）的$Q、K、V$来执行一个$Attention$，使用不同的线性映射得到多个$Q、K、V$来并行得执行$Attention$效果更佳，原因如下：

- 其增强了模型专注于不同信息的能力
- 为注意力层提供了多个“表示子空间”

**具体操作**：

对于每一个头，我们使用一套单独的权重矩阵$W_Q、W_K、W_V$，并且将其维度降至$d_{model}/H$

生成H个不同的注意力矩阵，将其拼接在一起

最后使用一个单独的权重矩阵$W^O$得到最终的注意力权重
$$
MutiHead(Q,K,V)=Concat(head_1,head_2,\cdots,head_h)W^O\\where\quad head_i=Attention(QW^Q_i,KW^K_i,VW^V_i)
$$

由于维度做了缩放，多头注意力的总代价和仅使用一个注意力的代价相近

**与卷积的关系**：

我们可以发现，多头注意力实际上与卷积有着异曲同工之妙

正如多个头可以注意不同的信息，不同的卷积核可以提取图像中不同的特征

同样，正如特征图多个通道内的信息冗余，多头注意力也存在着**信息冗余**

## 位置编码

[主要参考](https://blog.csdn.net/muyuu/article/details/110925334?ops_request_misc=&request_id=&biz_id=102&utm_term=%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-7-.first_rank_v2_pc_rank_v29)

### 为什么需要位置编码？

上文已经提到，Transformer是一种并行计算，为了让模型能够捕捉到序列的顺序关系，引入了位置编码，来获得单词之间的**相对距离**。

### 正余弦位置编码

$$
PE(pos,2i) = \sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}})\\
PE(pos,2i+1) = \cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})\\
$$

对于奇数位置使用余弦函数进行编码

对于偶数位置使用正弦函数进行编码

**注意**：这里的位置指的是一个词向量里数据的位置，pos指的才是单词在语句中的位置

例如某个单词在语句中的位置为Pos=5，$d_{model}=512$，则其位置编码向量为
$$
\left[\begin{array}{c}sin(\frac{5}{10000^{\frac{0}{128}}})\\cos(\frac{5}{10000^{\frac{0}{128}}})\\sin(\frac{5}{10000^{\frac{2}{128}}})\\cos(\frac{5}{10000^{\frac{2}{128}}})\\\vdots\\sin(\frac{5}{10000^{\frac{128}{128}}})\end{array}\right]
$$
可以看到，$2i、2i+1$仅仅决定使用的是$sin$还是$cos$，对于同一个$i$，内部是相同的

得到位置编码之后，将其与词向量相加，作为最终的输入

这里的直觉是，将位置编码添加到词向量，它们投影到$Q/K/V$并且进行点积时，会提供有意义的距离信息

### 为什么位置编码是有效的？

我们在小学二年级就学过三角函数的诱导公式：
$$
\sin(\alpha+\beta)=sin(\alpha)cos(\beta)+cos(\alpha)sin(\beta)\\
cos(\alpha+\beta)=cos(\alpha)cos(\beta)-sin(\alpha)sin(\beta)
$$


可以得到：
$$
PE(pos+k,2i)=PE(pos,2i)PE(k,2i+1)+PE(pos,2i+1)PE(k,2i)\\
PE(pos+k,2i+1)=PE(pos,2i+1)PE(k,2i+1)-PE(pos,2i)PE(k,2i)
$$
我们令$u(k)=PE(k,2i)、v(k)=PE(k,2i+1)$，得：
$$
\left[\begin{array}
{c}PE(pos+k,2i)\\ PE(pos+k,2i+1)
\end{array} \right]=
\left[\begin{array}
{c}v(k)&u(k)\\-u(k)&v(k)
\end{array} \right]
\left[\begin{array}
{c}PE(pos,2i)\\PE(pos,2i+1)
\end{array}\right]
$$
给定相对距离$k$，$PE(pos+k)$与$PE(pos)$之间具有**线性关系**

因此模型可以通过**绝对位置**的编码来更好地捕捉单词的**相对位置**关系

### 更多

毫无疑问，位置编码在整个Transformer中的作用是巨大的

没有位置编码的Tranformer就是一个巨型词袋

接下来让我们看看正余弦位置编码的局限

#### 相对距离的方向性

我们知道，**点积**可以表示相对距离，注意力机制中就使用点积作为打分函数来获取$Q、K$的相似度，让我们看看两个相对距离为k的位置编码的距离

对于$PE_{pos}$，令$c_i=\frac{1}{10000^{\frac{2i}{d}}}$：
$$
\begin{array}
PE_{pos}&=
\left[\begin{array}
{c}PE(pos,0)\\PE(pos,1)\\PE(pos,2)\\ \vdots\\PE(pos,d)
\end{array}\right]\\
&=
\left[\begin{array}
{c}sin(c_0pos)\\cos(c_0pos)\\sin(c_1pos)\\\vdots\\cos(c_{\frac{d}{2}-1}pos)
\end{array}\right]
\end{array}
$$
内积可得：
$$
\begin{array}PE_{pos}^TPE_{pos+k}&=
\sum_{i=0}^{\frac{d}{2}-1}{sin(c_ipos)PE(c_i(pos+k))+cos(c_ipos)cos(c_i(pos+k))}\\
&=\sum_{i=0}^{\frac{d}{2}-1}
{cos(c_i(pos+k-pos))}\\
&=\sum_{i=0}^{\frac{d}{2}-1}cos(c_ik)
\end{array}
$$
而我们知道，余弦函数是一个**偶函数**，因此正余弦编仅能捕捉到两单词之间的距离关系，而无法判断其距离关系

#### 自注意力对位置编码的影响

在Transfromer中，位置编码之后会进行自注意力的计算，公式如下：
$$
score(x_i)=\frac{(x_iW_Q)(x_iW_K)^T}{\sqrt{d}}=\frac{((x_i^{position}+x_i^{word})W_Q)((x_i^{position}+x_i^{word})W_K)^T}{\sqrt{d}}
$$

可以看到，经过自注意力的计算，模型实际上是无法保留单词之间的位置信息

那么Transformer是如何work的呢？

**题外话**：在bert中直接使用了Learned Position Embedding而非Sinusoidal position encoding

## 解码块

### Masked Multi-Head Attention

在机器翻译中，解码过程是一个顺序操作的过程，也就是当解码第$k$个特征向量时，我们只能看到第$k-1$及其之前的解码结果

而微调阶段，不使用MASK，这会导致预训练和微调数据的不统一，从而引入了一些人为误差，因此提出了[XLNet](https://arxiv.org/pdf/1906.08237.pdf)[^6]

### Endcoder-Decoder Attention

设计了一种解码块与编码块的交互模式

解码块最终的输入会生成$K，V$，输入给所有的解码器，而$Q$则是来自于Masked Multi-Head Attention



## 附录

[^1]:Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. 
[^2]:Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton,and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. 
[^3]: Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural Information Processing Systems
[^4]: Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Koray Kavukcuoglu. Neuralmachine translation in linear time
[^5]: Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning

[^6]: Zhilin Yang , Zihang Dai, Yiming Yang1, Jaime Carbonell , Ruslan Salakhutdinov , Quoc V. Le.XLNet: Generalized Autoregressive Pretraining for Language Understanding

