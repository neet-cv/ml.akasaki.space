---
title: Deep Retinex Decomposition for Low-Light Enhancement
authors: [ruomeng]
tags: [low-light]
---

> 该论文提出时间2018.7.20

## 大纲

> 对于Retinex方法，其为一种有效的低照度增强方法，它将观察到的图像分解为反射率和照度

大多数现有的Retinex方法都需要花费大量精力去设置分解的参数，以达到较好的效果，但是这样在实际场景中效果较差，而在这篇论文中，作者收集了一个低照度与正常光对比的低光数据集并基于该数据集的学习提出了一个Deep Retinex-net 

> Deep Retinex-net其中包括了一个 Decom-Net 用于分解 以及一个 Enhance-Net用于照度调节

> Decom-Net：（分解）在训练过程中**不考虑**分解后反射率和光照的基本事实，而是只学习两个关键的约束条件，低照度到正常图像共享的**一致反射率**以及照明的**平滑度**

> Enhance-Net：（增强）基于分解的基础，进行亮度增强

对于联合去噪，存在对于反射率的去噪操作，而在Retinex-net中是端到端可训练的，因此，对于分解的学习过程有助于亮度调整。

经过大量实验表明，作者的方法在视觉上的弱光增强获得了令人满意的效果，并且拥有图像分解的良好表现

<!--truncate-->

## 正文

### 1.低照度图像的影响

​    低照度图片下，会显著降低图像的可见性，丢失的细节和以及低对比度首先会对我们造成不愉快的视觉影响，而对于正常的计算机视觉系统的，照明不足的图片本身就不适用于正常的视觉系统，最终导致大量的性能损失。

### 2.引入低照度图像增强

​    为了使隐藏的细节可见，出现了大量的图像增强方法，以提高低照度图片的主观及客观质量。如直方图均衡化（HE）以及它的变体约束输出图像的直方图以满足某些约束，还有利用低照度图片和朦胧环境中的图像之间逆连接的De-hazing方法

> 另外一部分的低照度增强方法是基于Retinex的理论基础而提出的，其假设观察到的彩色图像可以被分解为反射率和照度，并以此扩展提出之后的各类经典方法。（1）SSR 作为一种早期的尝试，使用了高斯滤波器将光照图限制为平滑的。（2）MSRCR 通过多尺度高斯滤波器和颜色恢复扩展了SSR，并提出了一种利用亮度级误差测量来保持照明自然性的方法。（3）SRIE 使用加权变分模型同时估计反射率和照度。手动改变照度后，可以恢复目标结果。（4）LIME 只在结构先验下估计光照，并使用反射作为最终的增强结果。也有基于retinx的联合微光增强和噪声去除这两种方法。
>
> 随着深度神经网络的快速发展，CNN在低级别图像处理中得到了广泛的应用，包括super-resolution，rain removal等。Lore等人的使用堆叠稀疏去噪自动编码器来同时进行微光增强和降噪(LLNet)。

### 3.Retinex-Net的预计网络框架

![image-20210720205844362](https://gitee.com/ruomengawa/pic-go/raw/master/img/20210720205844.png)

> 增强过程分为三步:分解、调整和重构。在分解步骤中，子网Decom-Net将输入图像分解为反射率和照度。在接下
>
> 来的调整步骤中，一个基于编码器-解码器的增强网络将照明变亮。引入多尺度拼接，从多尺度角度调整光照。在这个步骤中，反射率上的噪声也被去除。最后，我们重建调整后的照明和反射率，以获得增强的结果

### 4. Retinex-Net for Low-Light Enhancement

$$
S = R\circ I
$$

S是源图像，R为反射率，I为照度，$ \circ$为逐元素相乘，其中R（反射率）描述了被捕获物体的内在属性，它被认为在任何亮度条件下都是一致的。照度表示物体上的各种亮度。在弱光图像上，它通常受到黑暗和不平衡照明分布的影响。（所以可以在结构图中发现R的normal和low的图片是一样的）

> 网络由三个步骤组成:分解、调整和重建。在分解步骤中，视网膜神经网络通过分解神经网络将输入图像分解成图像。它在训练阶段接收成对的弱光/正常光图像，而在测试阶段只接收弱光图像作为输入。在弱光/正常光图像具有相同的反射率和光照平滑度的约束下，Decom-Net学习以数据驱动的方式提取不同光照图像之间的一致R。在调整步骤中，使用增强网来照亮光照图。增强网络采用一个整体的编码器-解码器框架。使用多尺度拼接来保持大区域中照明与上下文信息的全局一致性，同时集中注意力调整局部分布。此外，如果需要，通常在弱光条件下出现的放大噪声将从反射率中去除。然后，我们在重建阶段通过逐元素乘法来组合调整后的照度和反射率。

### 5.Data-Driven Image Decomposition

对于分解一张图片，有一种方法是使用自己人工设置的限制条件直接在弱光输入图像上估计反射率和照度，但是！因为场景的不确定性，所以不容易设计适合各种场景的约束函数，所以最终作者选择使用了data-driven 的方式来解决分解问题。

于是，Decom-Net 出现了，**其每次接收成对的弱光/正常光图像，并且在低照度与正常图片共享相同的反射率前提下学习如何分解两者**，且虽然分解是用成对数据训练的，但它可以在测试阶段单独分解低照度输入。在训练过程中，不需要提供反射率和照度的常理条件。只有必要的知识，包括反射率的一致性和光照图的平滑度作为损失函数嵌入到网络中。因此，我们的网络的分解是从成对的低/正常光图像中自动学习的，并且本质上适合于描述不同光条件下图像之间的光变化。

值得注意的一点：虽然它和分解内在图像很相似，但是本质是不同的，所以在该任务中，我们的主要目的不应该是精确的获得实际的内在图像，而是一个很好的光线调节表示，我们需要的是让网络学习弱光图像与其增强图像之间的一致性成分

在实际使用中，作者先使用一个3 x 3卷积提取图像特征，而后再使用多个3 x 3卷积加上ReLU激活函数将RGB图像映射为反射和照明，而3 x 3卷积都是从特征空间投影R和I，并且使用sigmoid函数将 R 和 I 都投影在【0 , 1】的范围中

损失 $L$ 被分为三个部分: Reconstruction loss(重建损失) $L_{recon}$ ，Invariable reflectance loss（恒定反射损失）$L_{ir}$ , illumination smoothness loss(照明平滑度损失) $L_{is}$​ :
$$
L = L_{recon} + \lambda_{ir}L_{ir} +\lambda_{is}L_{is}
$$

> 其中$\lambda_{is}和$$\lambda_{it}$​​是平衡反射率一致性和照度平滑度的系数

基于Reflectance 图像都可以用相应照度映射来重建图像的假设，Reconstruction loss（重建损失）$L_{recon}$​被表述为：
$$
L_{recon}=\sum_{i=low,normal}\sum_{j=low,normal}\lambda_{i,j}||R_i\circ I_j-S_j||_1
$$
![image-20210722153835257](https://gitee.com/ruomengawa/pic-go/raw/master/img/20210722153835.png)

Invariable reflectance loss(恒定反射损失) $L_{ir}$ 被引入来限制反射率的一致性:
$$
L_{ir}=||R_{low}-R_{normal}||_1
$$

### 6.Structure-Aware Smoothness Loss

对于照度映射有一个基本假设是局部一致性和结构感知，换句话说，一个好的照度映射的解决方案应该是纹理细节平滑，同时仍然可以保留整体结构边界。

Total variation minimization (TV) 变差化最小化，其常常被用于最小化整个图像的梯度，经常被用作各种图像恢复任务的平滑度先验，然而，直接使用 TV 作为损失函数在图像具有强结构或亮度剧烈变化的区域会失败。这是由于无论该区域是文本细节区域还是强边界区域，光照图的梯度都是均匀减少的。换句话说，电视损失就是结构盲区化。照度模糊化，且会将顽固的黑色边缘留在反射图中

为了使得损失了解到图像结构，原始的TV函数应该被反射图映射加权，且最终的$L_{is}$应该被表述为

$$
L_{is}=\sum_{i=low,normal}||\nabla I_i\circ exp(-\lambda_g\nabla R_i)||
$$
限制由初始的照度映射加权，初始光照图是R，G，B通道中每个像素的最大值，在训练阶段我们可以同时更新照度和权重（反射率）

### 7.Multi-Scale Illumination Adjustment

> 照明增强网络采用编码器-解码器架构的整体框架。为了从分层的角度调整照明，我们引入了多尺度连接，编码器-解码器体系结构在大区域中获得上下文信息。输入图像被连续下采样到小尺度，在该尺度下网络可以获得大尺度照明分布的视角。这给网络带来了自适应调整的能力，而后，而后利用大尺度下的光照学习，上采样重建局部光照特征。通过元素求和，从下采样块到其对应的镜像上采样块引入跳跃连接，强制网络学习残差。

为了分层调整光照，即保持全局光照的一致性，同时适应不同的局部光照分布，引入的多尺度拼接，如果有 M 个递增的上采样块，每个上采样块提取一个拥有 C 个通道的特征图，我们通过最近邻插值调整这些不同比例的特征到最终比例，并将它们连接到一个拥有 C×M 通道的特征图。然后，通过1×1卷积层，级联特征被简化为 C 信道。遵循3×3卷积层来重建照度图

下采样块由步长为2的卷积层和ReLU组成。在上采样块中，使用了大小调整卷积层。它可以避免形成棋盘模式化的人工产物。调整大小卷积层由最近邻插值操作、步长为1的卷积层和ReLU组成。

增强函数中的损失 $L$ 由重建损失 $L_{recon}$ 和照明平滑度损失 $L_{is}$ 组成， $L_{recon}$​ 目的是制造正常光 $\hat{S}$ 其公式为
$$
L_{recon}=||R_{low}\circ\hat{I}-S_{normal}||_1
$$

### 8.Denoising on Reflectance

在分解过程中，作者对网络施加了若干约束，其中之一就是照度图的结构感知平滑性，当估计的照度图平滑时，图像细节将保留在反射图中，包括增强后的噪声，所以在网络中，我们还需要对反射图进行去噪操作，鉴于暗处噪声在分解过程中会根据亮度强度被放大，所以我们应该使用与光照相关的去噪方法。



