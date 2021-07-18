# 嵌入空间(Embedding)的理解

英文原文：[Neural Network Embeddings Explained](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)。

近年来，从计算机视觉到自然语言处理再到时间序列预测，神经网络、深度学习的应用越来越广泛。在深度学习的应用过程中，Embedding 这样一种将离散变量转变为连续向量的方式为神经网络在各方面的应用带来了极大的扩展。该技术目前主要有两种应用，NLP 中常用的 word embedding 以及用于类别数据的 entity embedding。

## Embedding和One Hot编码

