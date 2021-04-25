# `工具箱的`深度學習記事簿

本仓库的功能：`入门深度学习`、`参考常用代码`、`了解深度神经网络经典论文`。

<details>
<summary><strong>点击此处展开一堆P话</strong></summary>
“理解机器学习和深度学习的最佳方法是学以致用。”

Akasaki很菜，所以慢慢地学着，并在记事簿上写下笔记。Akasaki希望这个记事簿对自己很多有用，希望它对你也一样有用。这里包含了我从入门到依然在入门的过程中接触到的大部分知识。对于目录中的每一条，基本会有以下内容：

<p><strong>  1. 这个知识有什么用</strong></p>
<p><strong>  2. 这个知识如何使用或如何实现</strong></p>

请注意，本仓库中的大部分内容都在互联网上或各种书籍上能找得到。我所做的只是夹带私货地把它们串联在一起。

</details>

**这个仓库的部分文稿包含了LaTex公式**。我在每一个包含公式的文档头加入了mathjax相关的引用。如果你希望正常阅读，**请挂梯子访问[本仓库的网页版](https://ml.akasaki.space)**。如果你的网络条件导致了公式加载失败，请克隆本仓库到本地，并使用typora等markdown编辑器阅读。

## 目录

目录的使用方法：**点击就可以跳转到相关页面**。其中前面打了对勾的是已经可供阅读的部分，未打勾的是还没有写或正在写的部分。本仓库在宣布`archive`之前会保持更新。

如果你在阅读的过程中遇到了不明白的名词，**为了防止出现递归式学习，可以参照[附录-常见词汇](./appendix/similar-vocabularies.md)了解相关词汇的释义。**

### 第零章：在开始之前

- [x] [毁灭网友的人工智能](./ch0/ai-that-destroying-netizens.md)
- [x] [选框架比陪你女朋友逛超市还难的朋友可以点进来看看](./ch0/nn-and-frameworks.md)
- [x] [骗麻麻给你买一块显卡的理由（可不阅读）](./ch0/configure-gpu.md)
- [x] [安装轮子和管理环境](./ch0/create-new-environment-using-conda.md)

### 第一章：聊一些基础基础小知识

- [x] [试着读取、操作和显示数据](./ch1/operate-on-data.md)
- [x] [自动求梯度](./ch1/automatic-gradient.md)
- [x] [机器学习的HelloWorld：线性回归](./ch1/linear-regression.md)，以及[线性回归的代码实现](./ch1/linear-regression-code.md)
- [x] [softmax 回归](./ch1/softmax-regression.md)，以及[代码实现](./ch1/softmax-regression-code.md)
- [x] [多层感知机](./ch1/multilayer-perceptron.md)，以及[代码实现]()
- [ ] [模型的选择，欠拟合和过拟合]()
- [ ] [权重衰减]()
- [ ] [丢弃法]()
- [ ] [正向传播和反向传播]()

### 第二章上：卷积神经网络及其要素

- [x] [卷积神经网络和卷积操作](./ch2p1/convolutional-nn-and-ops.md)
- [ ] [二维卷积层]()
- [ ] [填充和步幅]()
- [ ] [多输入通道和多输出通道]()
- [ ] [池化层]()
- [ ] [批量归一化]()

### 第二章下：经典卷积神经网络

- [x] [第一个卷积神经网络(LeNet)](./ch2p2/LeNet.md)、相应[论文](./ch2p2/lecun-01a.pdf)以及[代码](./ch2p2/LeNet-code.md)
- [ ] [更深的卷积的神经网络(AlexNet)]()、相应[论文](./ch2p2/NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf)以及[代码]()
- [ ] [可复用的的网络单元(VGG)]()、相应[论文](./ch2p2/1409.1556VGG.pdf)以及[代码]()
- [ ] [带有残差的网络(ResNet)]()、相应[论文](./ch2p2/1512.03385ResNet.pdf)以及[代码]()
- [ ] [网络中的网络。(NiN)]()
- [ ] [含有并行连接的网络(GoogLeNet)]()
- [ ] [稠密链接网络(DenseNet)]()

### 第三章：循环神经网络

- [ ] [呃，语言模型是啥]()
- [ ] [循环神经网络！]()，以及[代码实现]()
- [ ] [一个语言模型的数据集（歌词）]()
- [ ] [通过时间反向传播]()
- [ ] [门控制循环单元(GRU)]()
- [ ] [长短期记忆(LSTM)]()
- [ ] [深度循环神经网络]()
- [ ] [双向循环神经网络]()

### 第四章：计算机视觉

- [x] [深度学习之于计算机视觉](./ch5/deep-learning-for-computer-vision.md)
- [x] [图像增广和微调](./ch5/image-augmentation.md)
- [ ] [目标检测中的边界框和锚框]()
- [ ] [多尺度目标检测]()
- [ ] [目标检测数据集（皮卡丘）]()
- [ ] [单发多框检测(SSD)]()
- [ ] [区域卷积神经网络(R-CNN)系列]()
- [ ] [语义分割和数据集]()
- [ ] [全卷积网络(FCN)]()
- [ ] [样式迁移]()
- [ ] [实战 Sifar-10]()，以及[实战 ImageNet-Dogs]()

### 附录

- [x] [常见的词汇](./appendix/similar-vocabularies.md)
- [ ] [快速配置环境以及安装包](./appendix/quick-envs-and-packages.md)
- [x] [工地matplotlib](./appendix/introducing-matplotlib.md)
- [x] [激活函数们](./appendix/activation-functions.md)
- [x] [总是用到的代码块](./appendix/similar-codeblocks.md)
- [ ] [常见的包]()
- [x] [工具箱？](./appendix/who-is-akasaki-toolbox.md)

### 第-1章：数字信号处理（DSP）

- [x] [何为DSP，为何DSP](./ch-1/about-dsp.md)
- [x] [周期信号](./ch-1/periodic-signal.md)
- [ ] [谐波]()
- [ ] [非周期信号]()
- [ ] [噪声]()
- [ ] [自相关]()
- [ ] [离散余弦变换]()
- [ ] [离散傅里叶变换]()
- [ ] [滤波与卷积]()
- [ ] [微分与积分]()
- [ ] [LTI系统]()
- [ ] [调制和采样]()

### 第-2章：TensorFlow编程策略

- [x] [TensorFLow编程策略](./ch-2/tensorflow-strategy.md)
- [ ] 让数据变得能够训练


---

## 捐助

如果这个仓库有用的话请在[github](https://github.com/visualDust/talkischeap)为[本仓库](https://github.com/visualDust/talkischeap)施舍一个star，这便是最好的捐助，是我继续写笔记的动力。

![_](https://jwenjian-visitor-badge-5.glitch.me/badge?page_id=VisualDust.anything)
![_](https://img.shields.io/github/stars/VisualDust/talkischeap.svg?style=flat)
![_](https://img.shields.io/github/license/visualdust/talkischeap.svg?style=flat&label=license&message=notspecified)