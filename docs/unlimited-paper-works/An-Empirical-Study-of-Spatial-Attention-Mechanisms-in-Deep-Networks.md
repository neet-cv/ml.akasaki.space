# An Empirical Study of Spatial Attention Mechanisms in Deep Networks

### 这篇笔记的写作者是[VisualDust](https://github.com/visualDust)。

这是一篇关于综述论文的解读。[原论文（An Empirical Study of Spatial Attention Mechanisms in Deep Networks）](https://arxiv.org/abs/1904.05873)。

摘要：

> Attention mechanisms have become a popular component in deep neural networks, yet there has been little examination of how different influencing factors and methods for computing attention from these factors affect performance. Toward a better general understanding of attention mechanisms, we present an empirical study that ablates various spatial attention elements within a generalized attention formulation, encompassing the dominant Transformer attention as well as the prevalent deformable convolution and dynamic convolution modules. Conducted on a variety of applications, the study yields significant findings about spatial attention in deep networks, some of which run counter to conventional understanding. For example, we find that the query and key content comparison in Transformer attention is negligible for self-attention, but vital for encoder-decoder attention. A proper combination of deformable convolution with key content only saliency achieves the best accuracy-efficiency tradeoff in self-attention. Our results suggest that there exists much room for improvement in the design of attention mechanisms.