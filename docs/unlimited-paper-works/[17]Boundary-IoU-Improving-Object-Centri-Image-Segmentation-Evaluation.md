# Boundary IoU: Improving Object-Centric Image Segmentation Evaluation

### 这篇笔记的写作者是[AsTheStarsFall](https://github.com/asthestarsfalll)。

> 论文名称：[Boundary IoU: Improving Object-Centric Image Segmentation Evaluation](https://arxiv.org/abs/2103.16562)
>
> 作者：Bowen Cheng，Ross Girshick，Piotr Dollár，Alexander C. Berg，Alexander Kirillov
>
> Code：https://github.com/bowenc0221/boundary-iou-api



写在前面：

​	**正如它的名字，Boundary IoU就是边界轮廓之间的IoU。**

​	重点为3.4节、5.1节，其他基本都是对比实验。



# 摘要

- 提出了一种新的基于边界质量的分割评价方法——Boundary IoU；
- Boundary IoU对大对象的边界误差比标准掩码IoU测量明显更敏感，并且不会过分惩罚较小对象的误差；
- 比其他方法更适合作为评价分割的指标。

# 介绍

- 对于分割任务，不同的评估指标对不同类型错误的敏感性不同，网络可以轻易解决对应敏感的类型，而其他错误类型的效果则不尽人意；

- mask的边界质量是图像分割的一个重要指标，各种下游任务直接受益于更精确的目标分割；

- 目前的分割网络的预测不够保真，边缘也很粗糙，**这种情况说明目前的评估指标可能对目标边界的预测误差具有有限的敏感性**；

  ![image-20210508214206239](https://gitee.com/Thedeadleaf/images/raw/master/20210508214210.png)

- 在大量的论文中，AP最高可达到八九十，而很少有论文会提及他们mask的边界质量。

- 对于实例分割，本文提出**Boundary Average Precision** (Boundary AP)，对于全景分割，提出**Boundary Panop-tic Quality** (Boundary PQ)。

# 相关指标

各种相关指标如下：

![image-20210508222750295](https://gitee.com/Thedeadleaf/images/raw/master/20210508223158.png)

首先解释几个名词：

1. 对称（Symmetric）：GT（GroundTruth）和Pred（prediction）的交换是否改变测量值

2. 倾向（Preference）：衡量方法是否偏向某一类型的预测。

3. 不灵敏度（Insensitivity）：测量不太敏感的误差类型。

4. 三分图（Trimap）：对给定图像的一种粗略划分将给定图像划分为前景、背景和待求未知区域。

   <img src="https://gitee.com/Thedeadleaf/images/raw/master/20210509151808.png" alt="img" />

5. Mask-based Measure：考虑物体的所有像素

6. Boundary-based Measure：衡量预测边界的分割质量，不同于Mask-based Measure，该方法只评估边界及其邻近的像素。

7. d：边界窄带的像素宽度

通过分析各种相关指标的缺点，我们得出Boundary IoU应该拥有的特性：**同时考虑分类、定位和分割质量。**

## Mask IoU和Pixel Accuracy

所有像素对指标的贡献都是相同的，而物体内部的像素呈二次型增长，其边界仅会线性增长，因此**对较大物体的边界不够敏感**。

Mask IoU计算方式示意图：

<img src="https://gitee.com/Thedeadleaf/images/raw/master/20210510161356.png" alt="image-20210510161350211" />

## Trimap IoU

基于边界的分割指标，其计算距离GT和pred边界d像素窄带内的IoU，计算方式示意图如下（方便起见，简化为矩形且只显示边界部分）：

<img src="https://gitee.com/Thedeadleaf/images/raw/master/20210509163907.png" alt="image-20210509163853163" />

**需要注意分母的**$G_d\cap G$。

## Feature Measure

F-Measure最初被提出用于边缘检测，但它也被用于评价分割质量。在最初的公式中，使用二分图匹配来进行计算，对于高分辨率的图像来说计算成本很大；因此提出了一种允许重复匹配的近似算法，**precision为pred轮廓中 \ 距离GT轮廓中像素 \ 在d个像素以内的 \ 像素 \ 所占pred的比例**（已断句），recall同理。不是很理解，原文如下：

![image-20210510151147870](https://gitee.com/Thedeadleaf/images/raw/master/20210510151207.png)

Precision和Recall计算方式示意图如下（可能）：

<img src="https://gitee.com/Thedeadleaf/images/raw/master/20210510153516.png" alt="image-20210510152547915" />

## Boundary  IoU

Boundary IoU对大物体边界误差更加敏感，并且不会过分惩罚小物体。

直观上就是GT和Pred轮廓的交集除以并集，但是**这里的轮廓是在对象内部的**$G_d、P_d$，不包括在对象外面的部分，详细请看9.1。

虽然看起来和Trimap IoU很相似，但个人认为它是Mask IoU的边界升级版本，去除了对象内部巨量像素对整体的影响（见5.1Mask IoU的分析），使其拥有更优秀的性质。 

完整的论文中给出的示意图如下：

![image-20210510153535293](https://gitee.com/Thedeadleaf/images/raw/master/20210510154511.png)

我画的：

<img src="https://gitee.com/Thedeadleaf/images/raw/master/20210510154514.png" alt="image-20210510154509338" />

# 敏感性分析

为了进行系统的比较，本文对GT进行处理形成伪预测，通过**模拟**不同的误差类型来尽可能的模拟真实误差类型。

## 尺度误差

通过对GT进行膨胀和腐蚀操作，误差严重程度由运算核半径控制。

![image-20210509185608432](https://gitee.com/Thedeadleaf/images/raw/master/20210509185613.png)

## 边界定位误差

将随机高斯噪声添加到GT上每一个多边形顶点的**坐标**上，误差严重程度由高斯噪声的标准差确定。

![image-20210509185545908](https://gitee.com/Thedeadleaf/images/raw/master/20210509185617.png)

## 物体定位误差

将GT中的对象随机偏移一些像素，误差严重程度由位移像素长度控制。

![image-20210509185530435](https://gitee.com/Thedeadleaf/images/raw/master/20210509185622.png)

## 边界近似误差

利用Sharply的简化公式来删除多边形顶点，同时保持简化多边形尽可能接近原始图像，误差严重程度由函数的容错参数控制。

![image-20210509185108649](https://gitee.com/Thedeadleaf/images/raw/master/20210509185624.png)

## 内部掩码错误

向GT中添加随机性形状的孔，虽然这种误差类型并不常见，但是本文将其包含进来，用以评估内部掩膜误差的影响。

![image-20210509185508685](https://gitee.com/Thedeadleaf/images/raw/master/20210509185630.png)

## 实现细节

**数据集**：作者从LVIS V0.5验证集中随机抽取实例掩码，因为该数据集拥有高质量的注释。

![](https://gitee.com/Thedeadleaf/images/raw/master/20210509190212.png)

**实现过程**：通过改变误差类型和误差的严重程度，记录每种类型的平均值和标准差，此外，还通过划分不同的区域，来比较对不同大小物体的指标评价。

其中d设置为图像对角线的2%。

# 现有方法分析

## Mask IoU

### 理论分析

**尺度不变性**（自己取的）：即对于一个**固定**的Mask IoU值，分割对象面积越大，则其错误像素越多，二者之间的变化关系成正比，其比例即为Mask IoU的值。

**惩罚差异性**（自己取的）：然而，当缩放一个对象时，内部像素数量呈二次增长，边界像素仅为线性增长，二者不同的增长率导致Mask IoU容忍更大的对象边界上的更多错误分类。

### 实证分析

**尺度不变性**基于一个假设，即GT标注中的边界误差也随着对象的大小而增长。

然而已有研究表明，不论物体大小，被不同标注器标记的同一个对象的两个轮廓之间的像素距离很少超过图像对角线的1%。（就叫它**标注相似性**吧）

本文通过研究LVIS提供的双标注图像来证实这一点，如下：

![image-20210509201406961](https://gitee.com/Thedeadleaf/images/raw/master/20210509203230.png)

其中冰箱的面积是机翼面积的100倍，但在相同分辨率的区域内，注释之间的差异在视觉上十分相似。

两者的两个轮廓的Mask IoU分别为0.97,0.81，而它们的Boundary IoU则更为接近，分别为0.87，0.81。说明Mask IoU**对小尺寸图片的“惩罚”更大**。

**实验**：通过严重程度相同的膨胀/腐蚀来模拟**尺度误差**，其显著降低了小物体的Mask IoU，而Mask IoU随物体大小的增加而增加，见下图：

![image-20210510093853486](https://gitee.com/Thedeadleaf/images/raw/master/20210510093856.png)

### 总结

- Mask IoU的主要不足在于对大物体边界的不敏感性。
- 相比之下，Boundary IoU更注重物体的边界。

## Trimap IoU

Trimap IoU是不对称的，交换GT和Pred将会得到不同的值。下图显示了其更倾向于比GT更大的pred：

![image-20210510095821668](https://gitee.com/Thedeadleaf/images/raw/master/20210510095941.png)

可以看到：

- 不论膨胀的严重程度是多少，其值总会大于某个正值，对小物体的“惩罚”依然过大。
- 腐蚀则会下降到零。

简单的证明：

![image-20210510165235885](https://gitee.com/Thedeadleaf/images/raw/master/20210511114830.png)

蓝色部分为pseudo-predictions （伪预测），红色方框为GT轮廓，可以看到，当pseudo-predictions 完全包含了GT时，其值不会再改变

同理，当伪预测完全被GT所包含，分子为0，最终值为0。

## F-measure

F-measure完全忽略了小的轮廓误差，但是表现效果很差，会在很短的严重程度中快速下降到0：

![image-20210510170006064](https://gitee.com/Thedeadleaf/images/raw/master/20210511114828.png)

## 总结

综上可知，F-measure和Trimap IoU都不能代替Mask IoU，而Mask IoU也有着不能忽视的缺陷，因此，本文提出Boundary IoU。

# Boundary IoU

## 公式

一个简化的IoU公式
$$
IoU = \frac{G_d\cap P_d}{G_d\cup P_d}
$$
该公式直接使用$G_d、P_d$,丢失了边缘的尖锐部分的信息

Boundary IoU公式如下：
$$
Boudary-IoU(G,P)=\frac{|(G_d\cap G)\cap(P_d\cap P)|}{|(G_d\cap G)\cup(P_d\cap P)|}
$$
其中参数d控制了测量的灵敏性，当d足够大时，Boundary IoU就相当于Mask IoU;若使用较小的d，Boundary IoU则会忽略内部像素，使其对边界像素更加敏感。

此外，对于较小的对象，Boundary IoU十分接近甚至等价于Mask IoU，这主要取决于参数d。

## Mask IoU vs Boundary IoU：敏感性分析

本文对比了Mask IoU和Boundary IoU在面积大于$96^2$的物体的不同误差类型下的表现：

![image-20210510173824215](https://gitee.com/Thedeadleaf/images/raw/master/20210510181123.png)

![image-20210510173839905](https://gitee.com/Thedeadleaf/images/raw/master/20210510181124.png)

对于每种误差类型，Boundary IoU都能更好的利用0-1的范围

使用的固定的误差严重程度，对大小不同的对象使用伪预测，以$16^2$为增量划分区域，二者表现如下：

![image-20210510181102929](https://gitee.com/Thedeadleaf/images/raw/master/20210510181127.png)

![image-20210510181118302](https://gitee.com/Thedeadleaf/images/raw/master/20210510181129.png)

可以看到：

- 对于较大的对象，Boundary IoU在相同严重程度下保持平缓，而Mask IoU则明显的偏向于大物体；
- 对于较小的对象，二者拥有相似的指标，说明他们都没有对其进行过度惩罚。

## Boundary IoU vs  Trimap IoU

二者具有一定的相似性，Boundary IoU将Pred和GT边缘上的像素都考虑了进来，这个简单的改进改变了Trimap IoU两点不足，一是不对称，二见4.2。

## Boundary IoU vs F-measure

F-measure对轮廓之间使用了硬预测——如果轮廓之间的像素在距离d内那么Precision和Recall都是完美的，然而当它们都位于d之外，则不会发生任何匹配（见4.3 ，其值会很快的降为0）。

而Boundary IoU使用一种软分割，变化平缓。

在附录中将会进行详细分析。

## 像素距离参数d

上文提过，当d足够大时，Boundary IoU等价于Mask IoU，当d过小，Boundary IoU则会出现严重惩罚的情况。

为了选择合适的参数d，本文在COCO和ASE20K两个数据集（它们拥有相似的分辨率）上进行实验，发现当d为图像**对角线的2%（大约为15个像素）**时，两数据集的Boundary IoU的中位数超过0.9。

对于Cityscapes中更大分辨率的图像，作者也建议使用相同的像素距离（15个左右），设置d为对角线的0.5%

对于其他数据集，作者建议考虑两个因素（**没看懂**：

1. 将注释一致性将下界设为d
2. D应根据当前方法的性能选择，并随着性能的提高而降低。

## Boundary IoU的局限

Boundary IoU不评估距离轮廓超过d的像素，例如一个圆形Mask和一个环形Mask：

![image-20210510190200706](https://gitee.com/Thedeadleaf/images/raw/master/20210511114822.png)

显然，其Boundary Iou值极高为1

为了惩罚这种情况，作者建议组合Boundary IoU和Mask IoU，并取他们的最小值。

此外，在实验中还发现，99.9%的情况Boundary IoU都是小于等于Mask IoU的，极少数情况如上图会出现Boundary IoU大于Mask IoU。

# 应用

如上文所说，作者将两种IoU组合，取其最小。

## Boundary AP for instance segmentation

实例分割任务的目标是用像素级掩码描绘每个对象，其评估指标是同时评估多个方面，如分类、定位和分割质量。

本文通过（Synthetic predictions，Synthetic，综合的；合成的，人造的，结合上下文个人感觉应该取“人造”之意） 合成预测与真实模型来进行实验。

### 合成预测

> 综合预测允许我们单独的评估分割质量。

- **具体方法**：

  使用COCO数据集，将GT缩小为28X28的连续值掩码，使用双线性插值upscale it back，最后将其二值化。如下图所示

  ![image-20210510230301360](https://gitee.com/Thedeadleaf/images/raw/master/20210510230303.png)

  这种合成Mask十分接近GT，但这种差异随着物体大小的增大而增大，因此越大的物体经过处理后的IoU值应该越低。

  ![image-20210510223226154](https://gitee.com/Thedeadleaf/images/raw/master/20210511122838.png)

  下标表示物体的大小，可以看到，对于越大的物体，Boundary IoU的值越低，而Mask IoU的值则维持在高水平，**这进一步显示了Boundary IoU对于大物体边界的敏感性**。

- 实验结果：在Mask RCNN、PointRend、以及BMask RCNN模型上进行实验，结果如下：

  ![image-20210510224102719](https://gitee.com/Thedeadleaf/images/raw/master/20210511122836.png)

  ![image-20210510224120918](https://gitee.com/Thedeadleaf/images/raw/master/20210511122835.png)

  众所周知，Mask RCNN对大物体的分割表现不尽人意（我不知道），从上表可以看出Boundary Ap的优越性

  此外，上表还证明了相较于BMask RCNN，PointRend对较大对象的表现更好。

  ![image-20210510224713604](https://gitee.com/Thedeadleaf/images/raw/master/20210511122833.png)

  上表显示了更深的主干网络并不能带来分割质量的显著提升。

### 真实预测

> 利用现有的分割模型得到的真实预测进一步实验，可以进一步了解Boundary IoU在实例分割任务各个方面的表现。

- **具体方法**：

  为了将分割质量与分类和定位错误分离开，作者为这些方法提供了Ground Truth Box，并为其分配随机置信度。

- **实验结果**：

  模型在COCO数据集上训练，在LVIS v0.5上验证

  ![image-20210511115502161](https://gitee.com/Thedeadleaf/images/raw/master/20210511115505.png)

  模型在Cityscapes上训练和验证

  ![image-20210511115655795](https://gitee.com/Thedeadleaf/images/raw/master/20210511115657.png)

## Boundary  PQ

下图为标准PQ的公式

<img src="https://gitee.com/Thedeadleaf/images/raw/master/20210511115040.png" alt="image-20210511115032369" />

将其中的Mask IoU替换为Mask IoU与Boundary IoU的组合，取其最小值。

### 合成预测

<img src="https://gitee.com/Thedeadleaf/images/raw/master/20210511122826.png" alt="image-20210511120047274" />

### 真实预测

![image-20210511120131765](https://gitee.com/Thedeadleaf/images/raw/master/20210511122824.png)

# 总结

​		不同于Mask IoU，Boundary IoU提供了一个明确的，定量的梯度，奖励改善边界分割质量。作者希望Boundary IoU可以鼓励更多人开发高保真Mask预测新方法。此外，Boundary  IoU允许对复杂的任务(如实例和全景分割)的分割相关错误进行更细粒度的分析。在性能分析工具(如TIDE[2])中结合度量可以更好地洞察实例分段模型的特定错误类型。（**直接翻译的**）

# 补充

## $G_d$和$G_d\cap G$

![image-20210511121230297](https://gitee.com/Thedeadleaf/images/raw/master/20210511121300.png)

## 代码复现

对于二分类图像的Boundary Iou

```python
# 将二值Mask转化为Boundary mask
def mask_to_boundary(mask, dilation_ratio=0.01):
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    # 将mask使用0填充一圈，防止dilation为1时
    new_mask = cv2.copyMakeBorder(
        mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    # 对mask进行腐蚀操作
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
    # G_d intersects G
    return mask - mask_erode

def boundary_iou(mask, pred):
    intersect = mask*pred
    ite = np.sum(intersect == 1)
    un = mask+pred
    union = np.sum(un >= 1)
    return ite/union
```

![image-20210519091807762](https://gitee.com/Thedeadleaf/images/raw/master/20210519091830.png)

![image-20210519091815466](https://gitee.com/Thedeadleaf/images/raw/master/20210519091840.png)

![image-20210519091826066](https://gitee.com/Thedeadleaf/images/raw/master/20210519091833.png)
