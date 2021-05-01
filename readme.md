# `工具箱的`深度學習記事簿

<details>
<summary><strong>点击此处展开一堆P话</strong></summary>
“理解机器学习和深度学习的最佳方法是学以致用。”

Akasaki很菜，所以慢慢地学着，并在记事簿上写下笔记。Akasaki希望这个记事簿对自己很多有用，希望它对你也一样有用。这里包含了我从入门到依然在入门的过程中接触到的大部分知识以及思考的过程。对于目录中的每一条，基本会有以下内容：

<p><strong>  1. 这个知识有什么用</strong></p>
<p><strong>  2. 这个知识如何使用或如何实现</strong></p>

希望这份笔记对你也有用。如果真的有用，请留下一个star。

</details>

- 本仓库的功能：`入门深度学习`、`参考常用代码`、`了解深度神经网络经典论文`。请注意，本仓库后半部分主要讨论计算机视觉领域的基于深度学习的方法。
- 本仓库中的第一章直接引用[d2l-zh](https://github.com/d2l-ai/d2l-zh)原书内容，并稍作修改；后面是自己学习、阅读书籍、阅读书籍、阅读博文后整理的内容，很多内容在各大书籍和各大搜索引擎都能找到，我所做的是用自己的想法整理或书写这些内容。也有很多内容是我自己的理解以及夹带的私货。
- 本仓库的很多代码主要使用[Tensorflow](https://www.tensorflow.org/)等实现。也许等我写完后[我的朋友们](https://github.com/PaperFormulaIV)会帮我翻译出一份pytorch版本。
- 这个仓库的部分文稿**包含了LaTex公式**。我在每一个包含公式的文档头加入了mathjax相关的引用。如果你希望正常阅读，**请挂梯子访问[本仓库的网页版](https://ml.akasaki.space)**。如果你的网络条件导致了公式加载失败，请克隆本仓库到本地，并使用typora等markdown编辑器阅读。

##  目录

目录的使用方法：**点击就可以跳转到相关页面**。其中前面打了对勾的是已经可供阅读的部分，未打勾的是还没有写或正在写的部分。本仓库在宣布`archive`之前会保持更新。

>  如果你在阅读的过程中遇到了不明白的名词，**为了防止出现递归式学习，可以参照[附录-常见词汇](./appendix/similar-vocabularies.md)了解相关词汇的释义。**

### 第零章：在开始之前

诶**等等**！你确定你真的想碰这个方向吗？

- [x] [01. 毁灭网友的人工智能](./ch0/ai-that-destroying-netizens.md)
- [x] [02. 选框架比陪女朋友逛超市还难](./ch0/nn-and-frameworks.md)
- [x] [03. 剁手买一块显卡的理由（可不阅读）](./ch0/configure-gpu.md)
- [x] [04. 体验痛苦：安装轮子和管理环境](./ch0/create-new-environment-using-conda.md)

---

### 第一章上：这HelloWorld有点长啊

我猜你见过一些诸如深度学习的计算机视觉、深度学习的自然语言处理等名词，但是你肯定没看到过深度学习自己单独出现。没错，**深度学习是一种方法，往往用来解决各种问题。所以我们首先要了解方法的基本思想，才能看如何用它解决问题**。所以这一章我们主要了解最基本的机器学习任务线性回归以及其他的一些基础知识。看**完本章也许初学者会不知所以然**。但是请继续看下去，你就会发现这些东西的实际用处。

- [x] [01. 试着读取、操作和显示数据](./ch1p1/operate-on-data.md)
- [x] [02. 自动求梯度](./ch1p1/automatic-gradient.md)
- [x] [03. 机器学习的HelloWorld：线性回归](./ch1p1/linear-regression.md)，以及[线性回归的代码实现](./ch1p1/linear-regression-code.md)
- [x] [04. softmax 回归](./ch1p1/softmax-regression.md)，以及[代码实现](./ch1p1/softmax-regression-code.md)

---

### 第一章下：深度学习基础——多层感知机

终于有点“人工神经网络”的样子了！从多层感知机开始，你将正式步入搭建人工神经网络的大门。这一部分看上去比上一部分有用多了。**多层感知机及其方法是人工神经网络的基本构想**，它们从结构上层层相连，就像是神经的连接方式；在多层感知机中，每个节点负责非常简单的计算。连在一起构成能力更强大的决策网络，这也像极了动物的神经系统。

- [x] [05. 多层感知机](./ch1p2/multilayer-perceptron.md)，以及[代码实现](./ch1p2/multilayer-perceptron-code.md)
- [x] [06. 模型的选择，欠拟合和过拟合](./ch1p2/underfit-and-overfit.md)
- [x] [07. 权重衰减](./ch1p2/weight-decay.md)
- [x] [08. 丢弃法](./ch1p2/dropout.md)
- [x] [09. 正向传播和反向传播](./ch1p2/forward-and-backprop.md)
- [x] [10. 数值稳定和模型初始化](./ch1p2/numerical-stability-and-initializing.md)
- [ ] [11. 使用多层感知机实现手写数字识别]()

---

### 第二章上：卷积神经网络及其要素

卷积神经网络，听上去似乎是一个很厉害的东西。在这里先抛出一个结论：**相较于全连接网络，卷积神经网络相对的进步在于卷积层结构和池化结构的引入**。也就是说，在全连接网络的基本思想上，塞进卷积和池化，就变成了一个“更厉害”的东西。所以在这半章，我们主要了解这两样东西：**卷积**和**池化**。

- [x] [01. 卷积神经网络和卷积操作](./ch2p1/convolutional-nn-and-ops.md)
- [ ] [02. 池化及其基本实现方法]()
- [ ] [03. 二维卷积层、池化层联动实例]()
- [ ] [04. 一些细节：填充、步长和感受野](./ch2p1/strides-padding-and-receptive-field.md)
- [ ] [05. 全卷积、以及对输入输出通道数的进一步了解]()
- [ ] [06. 批量归一化和用处]()
- [ ] [07. 使用卷积神经网络实现手写数字识别]()

---

### 第二章下：经典卷积神经网络

上半章对卷积神经网络的一些基础知识做了简单介绍，下半章主要以历史的眼光观察经典的卷积神经网络。从LeNet手写数字识别成为第一个卷积神经网络的应用开始，到后来卷积神经网络往越来越深，成为深度卷积神经网络，再到后来出现并行、残差等概念，它们逐渐变得越来越成熟。这半章我们会了解这些网络的大体思路和代码写法。当然，如果你有兴趣可以阅读论文本身。

- [x] [01. 第一个卷积神经网络：LeNet](./ch2p2/LeNet.md)、相应[论文](./ch2p2/lecun-01a.pdf)以及[代码](./ch2p2/LeNet-code.md)
- [ ] [02. 小插曲：Keras高级API]()
- [x] [03. 更深的卷积的神经网络：AlexNet](./ch2p2/AlexNet.md)、相应[论文](./ch2p2/NIPS-2012-imagenet-classification-with-deep-convolutional-neural-networks-Paper.pdf)以及[代码](./ch2p2/AlexNet-code.md)
- [ ] [04. 可复用的的网络单元：VGG]()、相应[论文](./ch2p2/1409.1556VGG.pdf)以及[代码]()
- [ ] [05. 带有残差的网络：ResNet]()、相应[论文](./ch2p2/1512.03385ResNet.pdf)以及[代码]()
- [ ] [06. 网络中的网络：NiN]()
- [ ] [07. 致敬LeNet：含有并行连接的网络GoogLeNet]()
- [ ] [08. 稠密链接网络：DenseNet]()

---

### 第三章上：谈一些计算机视觉的方向

深度学习是一种方法，它可以用来解决各类问题。但是很明显地球上所有的问题多到不能被枚举（怎么突然上升到了哲学高度），而上一章我们恰好聊了卷积神经网络，并且卷积神经网络近年来在计算机视觉领域有着很突出的表现。所以我们在这里我们阅读一些相关领域的综述论文，大致了解深度学习技术在计算机视觉领域的应用。

- [x] [01. 深度学习之于计算机视觉](./ch3p1/deep-learning-for-computer-vision.md)
- [ ] [02. 目标检测方法综述]()
- [x] [03. 语义分割方法综述](./ch3p1/overview-of-semantic-segmentation.md)以及[原综述论文](./ch3p1/A-Review-on-Deep-Learning-Techniques-Applied-to-Semantic-Segmentation.pdf)
- [x] [04. 样式迁移方法](./ch3p1/image-style-transfer.md)
- [x] [05. 不着边的插曲：图像增广和微调](./ch3p1/image-augmentation.md)

---

### 第三章下：尝试一些计算机视觉任务

君子动手不动口。

- [ ] [01. 小插曲：目标检测中的边界框和锚框]()
- [ ] [02. 多尺度目标检测]()
- [ ] [04. 单发多框检测(SSD)]()
- [ ] [05. 区域卷积神经网络(R-CNN)系列]()
- [ ] [06. 杠上Sifar-10]()
- [ ] [07. 尝试全卷积神经网络进行分割]()
- [ ] [08. 跑别人家的U^2Net]()

---

### 第四章：循环神经网络

- [ ] [01. 小插曲：语言模型]()
- [ ] [02. 循环神经网络]()，以及[代码实现]()
- [ ] [03. 使用歌词作为语言模型的数据集]()
- [ ] [04. 通过时间反向传播]()
- [ ] [05. 门控制循环单元(GRU)]()
- [ ] [06. 长短期记忆(LSTM)]()
- [ ] [07. 深度循环神经网络]()
- [ ] [08. 双向循环神经网络]()

---

### 附录

- [x] [01. 常见的词汇](./appendix/similar-vocabularies.md)
- [ ] [02. 快速配置环境以及安装包](./appendix/quick-envs-and-packages.md)
- [x] [03. 工地matplotlib](./appendix/introducing-matplotlib.md)
- [x] [04. 激活函数们](./appendix/activation-functions.md)
- [x] [05. 总是用到的代码块](./appendix/similar-codeblocks.md)
- [ ] [06. 常见的包]()
- [x] [07. 工具箱？](./appendix/who-is-akasaki-toolbox.md)

---

### 第-1章：TensorFlow编程策略

能翻到这里，说明你也应该在愁这些东西了。为什么这章会是-1章呢？因为这些内容都是专门针对Tensorflow而订制的。很明显在前面的正式章节中并没有谁希望过度依赖于某个框架。但是如果你开始正式使用Tensorflow，你一定会用上这些知识的。

- [x] [01. TF张量操作API第一部分](./ch-1/operation-on-tensors-1.md)
- [x] [02. TF张量操作API第二部分](./ch-1/operation-on-tensors-2.md)
- [x] [03. 对张量进行基本数学运算](./ch-1/operator-for-tensors.md)
- [x] [04. TensorFLow编程策略：计算图和模型表示](./ch-1/tensorflow-strategy.md)
- [ ] [05. 训练数据的格式]()
- [ ] [06. 模型持久化]()
- [ ] [07. 训练可视化（以Tensorboard为例）]()
- [ ] [08. 多卡、多设备加速计算]()

---

### 第-2章：数字信号处理（DSP）

这一章对入门的人来说似乎用处并不大（而我，还站在门外），所以我把它放在了第-2章。不过似乎入门入久了，就会开始好奇一些东西背后的原理。之前在卷积的章节中我们提过一嘴，“在TensorFlow官方文档中将卷积核称为‘过滤器(Filer) ’，卷积操作称为‘滤波操作’，因为它们本就是表达了一个意思。”说道这里你可能已经大概明白了，数字信号处理和卷积之前有很多隐隐约约的关系。**重申一遍，这一章真的不是必须看的，当你想看的时候再回来也不迟**。

- [x] [01. 何为DSP，为何DSP](./ch-2/about-dsp.md)
- [x] [02. 周期信号](./ch-2/periodic-signal.md)
- [ ] [03. 谐波]()
- [ ] [04. 非周期信号]()
- [ ] [05. 噪声]()
- [ ] [06. 自相关]()
- [ ] [07. 离散余弦变换]()
- [ ] [08. 离散傅里叶变换]()
- [ ] [09. 滤波与卷积]()
- [ ] [10. 微分与积分]()
- [ ] [11. LTI系统]()
- [ ] [11. 调制和采样]()


---

## 继续学习

当你看完本仓库所有内容时，恭喜你，你已经**算得上是入门了**。你在这里简要了解了深度神经网络对计算机视觉以及其他一些机器学习任务的深刻影响，**接下来你应该选一个你想继续深刻研究的方向**。

在你开始研究的初期，你可以找到一个听上去很感兴趣的方向，然后试着读以下该方向的综述，确定你是否真的喜欢它。一般情况下我会这样了解一个方向：

- [ ] 这里应该马上会有一篇短文

我还会继续开放用于下一步学习的仓库（在写了在写了），以便后续参考。

---

## 捐助

如果这个仓库有用的话请在[github](https://github.com/visualDust/talkischeap)为[本仓库](https://github.com/visualDust/talkischeap)施舍一个star，这便是最好的捐助，是我继续写笔记的动力。

![image-20210427212743443](src/readme/image-20210427212743443.png)

![_](https://jwenjian-visitor-badge-5.glitch.me/badge?page_id=VisualDust.anything)
![_](https://img.shields.io/github/stars/VisualDust/talkischeap.svg?style=flat)
![_](https://img.shields.io/github/license/visualdust/talkischeap.svg?style=flat&label=license&message=notspecified)

---

## 无尽模式

我在这里阅读论文，并写下笔记。

『`分割`、`解码器`、`上采样`、`20210429`』[The Devil is in the Decoder: Classification, Regression and GANs](./unlimited-paper-works/The-Devil-is-in-the-Decoder-Classification-Regression-and-GANs.md)以及[原论文](./unlimited-paper-works/The-Devil-is-in-the-Decoder-Classification-Regression-and-GANs.pdf)

『`恶意样本生成`、`adv learning`、`神经网络攻击`、`20210501`』[Threat of Adversarial Attacks on Deep Learning in Computer Vision: A Survey](./unlimited-paper-works/Threat-of-Adversarial-Attacks-on-Deep-Learning-in-Computer-Vision-A-Survey.md)以及[](./unlimited-paper-works/Threat-of-Adversarial-Attacks-on-Deep-Learning-in-Computer-Vision-A-Survey.pdf)



