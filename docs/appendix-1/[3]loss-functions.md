# 损失函数们

![img](./src/loss-functions/GettyImages-154932300.jpg)

> 损失函数（loss function）或代价函数（cost function）是将随机事件或其有关随机变量的取值映射为非负实数以表示该随机事件的“风险”或“损失”的函数。 在应用中，损失函数通常作为学习准则与优化问题相联系，即通过最小化损失函数求解和评估模型。

首先我们对三个名词进行说明：

- 损失函数（Loss function）用于定义单个训练样本与真实值之间的误差。

- 代价函数（Cost function）用于定义单个批次或整个训练集样本与真实值之间的误差。

- 目标函数（Objective function）则泛指任意可以被优化的函数。

在机器学习中，损失函数是代价函数的一部分，用于衡量模型所作出的预测离真实值（Ground Truth）之间的偏离程度；而代价函数则是目标函数的一种类型。通常，我们都会最小化目标函数，最常用的算法便是 **“梯度下降法”（Gradient Descent）**。

实际上，目标函数大都是针对某种任务提出的，并没有一种万能的损失函数能够适用于所有的机器学习任务，所以在这里我们需要知道每一种损失函数的优点和局限性，才能更好的利用它们去解决实际的问题。损失函数大致可分为两种：**回归损失**（针对**连续型**变量）和**分类损失**（针对**离散型**变量）。其中回归损失多用于目标检测等回归任务，分类损失多用于分割等密集预测任务。

## 回归损失（Regression Loss）

### $L_1$ Loss

也称为**Mean Absolute Error**，即平均绝对误差（MAE），它衡量的是预测值与真实值之间距离的平均误差幅度，作用范围为0到正无穷。
$$
L_1 = |f(x)-Y|
$$
导数：
$$
L_1' = \begin{cases}
-f'(x)& \text{f(x)<0}\\
f'(x)& \text{otherwise}
\end{cases}
$$


优点： 收敛速度快，能够对梯度给予合适的惩罚权重，而不是“一视同仁”，使梯度更新的方向可以更加精确。

缺点： 对异常值十分敏感，梯度更新的方向很容易受离群点所主导，不具备鲁棒性。

### $L_2$ Loss

也称为**Mean Squred Error**，即均方差（**MSE**），它衡量的是预测值与真实1值之间距离的平方和，作用范围同为0到正无穷。
$$
L_2 = |f(x)-Y|^2
$$
导数：
$$
L_2' = 2f'(x)(f(x)-Y)
$$
优点：对离群点（**Outliers**）或者异常值更具有鲁棒性。

缺点：由图可知其在0点处的导数不连续，使得求解效率低下，导致收敛速度慢；而对于较小的损失值，其梯度也同其他区间损失值的梯度一样大，所以不利于网络的学习。

###  Smooth $L_1$ Loss

即平滑的$L_1$损失（SLL），出自Fast RCNN [7]。SLL通过综合$L_1$和$L_2$损失的优点，在0点处附近采用了$L_2$损失中的平方函数，解决了$L_1$损失在0点处梯度不可导的问题，使其更加平滑易于收敛。此外，在|x|>1的区间上，它又采用了$L_1$损失中的线性函数，使得梯度能够快速下降。
$$
L_{1_{smooth}}(x) = \begin{cases}
0.5x^2& \text{|x|<1}\\
|x|-0.5& \text{otherwise}
\end{cases}
$$
导数：
$$
L_{1_{smooth}}' =\begin{cases}
x& \text{|x|<1}\\
-1& \text{x< -1}\\
1& \text{x>1}
\end{cases}
$$
L1损失的导数为常数，如果不及时调整学习率，那么当值过小时，会导致模型很难收敛到一个较高的精度，而是趋向于一个固定值附近波动。反过来，对于L2损失来说，由于在训练初期值较大时，其导数值也会相应较大，导致训练不稳定。最后，可以发现Smooth L1在训练初期输入数值较大时能够较为稳定在某一个数值，而在后期趋向于收敛时也能够加速梯度的回传，很好的解决了前面两者所存在的问题。

### IoU Loss

即交并比损失，出自UnitBox，由旷视科技于ACM2016首次提出。常规的$L_{\cdot}$型损失中，都是基于目标边界中的4个坐标点信息之间分别进行回归损失计算的。因此，这些边框信息之间是相互**独立**的。然而，直观上来看，这些边框信息之间必然是存在某种**相关性**的。

![image-20210718150907572](./src/loss-functions/image-20210718150907572.png)

上图中，绿色框代表Ground Truth，蓝色框代表Prediction。显然**重叠度**越高的预测框是越合理的。IoU Loss的核心是一个被称为IoU（Intersection over Union）的表达式：
$$
IoU = \frac{|B\cap B^{gt}|}{|B\cup B^{gt}|}
$$
该表达式是一种度量框的重叠程度的公式。为了解决IoU度量不可导的现象，引入了负Ln范数来间接计算IoU Loss：
$$
Loss_{IOU} = -Ln(IoU) = -Ln \frac{|B\cap B^{gt}|}{|B\cup B^{gt}|}
$$
在应用中，也出现了其他Iou的计算形式：
$$
Loss_{IOU} = 1- IoU = 1- \frac{|B\cap B^{gt}|}{|B\cup B^{gt}|}
$$
IoU损失将候选框的四个边界信息作为一个**整体**进行回归，从而实现准确、高效的定位，具有很好的**尺度不变性**（IoU的值总在$[0,1]$之间）。

### GIoU Loss

上文中的IoU Loss存在一个问题，就是IoU Loss只能衡量prediction和ground truth有重叠的情况：

![image-20210718170101661](./src/loss-functions/image-20210718170101661.png)

例在上图中，IoU Loss只能衡量图1中的情况。对于图2和图3中的情况，其Loss值是一样的。例如，当采用$Loss_{IOU} = -Ln(IoU) = -Ln \frac{|B\cap B^{gt}|}{|B\cup B^{gt}|}$的情况，由于两个框交集（IoU的分子）为0，上式的值永远为$+\infin$。再例如，当采用$Loss_{IOU} = 1- IoU = 1- \frac{|B\cap B^{gt}|}{|B\cup B^{gt}|}$的情况，由于两个框的交集（IoU的分子）为0，上式的值永远为$0$。这会导致出现梯度消失的现象，致使网络无法给出一个**优化的方向**。

也就是说，当两个框未出现交集时，IoU Loss将无法量化其距离造成的差距。在直观上，虽然并未出现交集，但两个框的距离较大时，Loss应该更大才对，这样有利于更好地更新梯度。

GIoU（Generalized Intersection over Union）即泛化的IoU（由斯坦福学者于CVPR2019的[论文](https://arxiv.org/abs/1902.09630)中提出），能够解决这个问题：

![image-20210718172024968](./src/loss-functions/image-20210718172024968.png)

在上图中，绿色的实线框表示ground truth，黑色的实线框表示prediction，黑色的虚线框表示**两个框的最小闭区域面积**（可通俗理解为同时包含了预测框和真实框的最小框的面积），记为$C$。则：
$$
GIoU = IoU - \frac{|C\setminus(B\cup B^{gt})|}{|C|}
$$
上式中$|C\setminus(B\cup B^{gt})|$表示$C$中除了$B$和$B^{gt}$的部分，$\setminus$是差集符号。

与IoU Loss的设计类似，GIoU Loss可以设计为：
$$
Loss_{GIoU} = 1-GIoU
$$
和IoU一样，GIoU也具有scale不敏感的特性，在两个框完全重合的情况下，$IoU = GIoU = 1$。不同的是，IoU的取值范围是$[0,1]$，而GIoU的取值是$[-1,1]$的对称区间。与IoU只关注重叠区域不同，**GIoU不仅关注重叠区域，还关注其他的非重合区域**，能更好的反映两者的重合度。

### DIoU Loss

DIoU即距离IoU损失，全称为Distance-IoU loss，由天津大学数学学院研究人员于AAAI2020所发表的这篇[论文](https://arxiv.org/abs/1911.08287)中首次提出。上面我们谈到GIoU通过引入最小闭合凸面来解决IoU无法对不重叠边框的优化问题。但是，其仍然存在两大局限性：

- 边框回归还不够精确
- 收敛速度缓慢

考虑下图这种情况，当目标框**完全包含**预测框时，此时GIoU**退化**为IoU。

![image-20210726184128613](./src/loss-functions/image-20210726184128613.png)

显然，我们希望的预测是最右边这种情况。因此，作者通过计算两个边框之间的**中心点归一化距离**，从而更好的优化这种情况。

下图表示的是GIoU损失（第一行）和DIoU损失（第二行）的一个训练过程收敛情况：

![image-20210726184756931](./src/loss-functions/image-20210726184756931.png)

其中绿色框为目标边框，黑色框为锚框，蓝色框和红色框则分别表示使用GIoU损失和DIoU损失所得到的预测框。可以发现，GIoU损失一般会增加预测框的大小使其能和目标框重叠，而DIoU损失则直接使目标框和预测框之间的中心点归一化距离最小，即让预测框的中心快速的向目标中心收敛。

现给出三种IoU的公式对比：
$$
IoU = \frac{|B\cap B^{gt}|}{|B\cup B^{gt}|}\\
GIoU = IoU - \frac{|C\setminus(B\cup B^{gt})|}{|C|}\\
DIoU = IoU - \frac{\rho^2(b,b^{gt})}{c^2}
$$
同样地，DIoU Loss的值也是$1-DIoU$。

对于DIoU来说，其惩罚项由两部分构成：分子为目标框和预测框中心点之间的欧式距离；分母为两个框最小外接矩形框的两个对角线距离：

![image-20210726190106028](./src/loss-functions/image-20210726190106028.png)

因此， 直接优化两个点之间的距离会使得模型**收敛得更快，同时又能够在两个边框不重叠的情况下给出一个优化的方向。**

### CIoU Loss

即完整IoU损失，全称为Complete IoU loss，与DIoU出自同一篇论文。上面我们提到GIoU存在两个缺陷，DIoU的提出解决了其实一个缺陷，即收敛速度的问题。而一个好的边框回归损失应该同时考虑三个重要的几何因素，即重叠面积（**Overlap area**）、中心点距离（C**entral point distance**）和高宽比（**Aspect ratio**）。GIoU考虑到了重叠面积的问题，DIoU考虑到了重叠面积和中心点距离的问题，CIoU则在此基础上进一步的考虑到了高宽比的问题。
$$
CIoU = IoU - \frac{\rho^2(b,b^{gt})}{c^2} + \alpha v
$$
上式为CIoU的计算公式，可以看出，其在DIoU的基础上加多了一个惩罚项$\alpha v$​。其中$\alpha$​为权重为正数的重叠面积平衡因子，在回归中被赋与更高的优先级，特别是在两个边框不重叠的情况下；而$v$​​则用于测量宽高比的一致性。

### F-EIoU Loss

Focal and Efficient IoU Loss是由华南理工大学学者最近提出的一篇关于目标检测损失函数的论文，文章主要的贡献是提升网络收敛速度和目标定位精度。目前检测任务的损失函数主要有两个缺点：

- 无法有效地描述边界框回归的目标，导致收敛速度慢以及回归结果不准确。
- 忽略了边界框回归中不平衡的问题。

F-EIou loss首先提出了一种有效的交并集（IOU）损失，它可以准确地测量边界框回归中的**重叠面积**、**中心点**和**边长**三个几何因素的差异：
$$
L_{EIoU} = L_{IoU} + L_{dis} + L_{asp}\\
 = 1 - IoU + \frac{\rho^2(b,b^{gt})}{c^2} + \frac{\rho^2(w,w^{gt})}{C_w^2} + \frac{\rho^2(h,h^{gt})}{C_h^2}
$$
其次，基于对有效样本挖掘问题（EEM）的探讨，提出了Focal loss的回归版本，以使回归过程中专注于高质量的锚框：
$$
L_f(x)=
\begin{cases}
-\frac{\alpha x^2 (2\ln(\beta x)-1)}{4} & 0<x\leq1;1/e\leq\beta\leq1\\
-\alpha\ln(\beta)x+C & x>1;1/2\leq\beta\leq1
\end{cases}
$$
最后，将以上两个部分结合起来得到Focal-EIou Loss：
$$
L_{Focal-EIoU} = \frac{\sum_{i=1}^n W_i\cdot L_{EIOU}}{\sum_{i=1}^n W_i}
$$
其中，通过加入每个batch的权重和来避免网络在早期训练阶段收敛慢的问题。

### CDIoU Loss

Control Distance IoU Loss是由同济大学学者提出的，文章的主要贡献是在几乎不增强计算量的前提下有效提升了边界框回归的精准度。目前检测领域主要两大问题：

- SOTA算法虽然有效但计算成本高。
- 边界框回归损失函数设计不够合理。

![image-20210726203705056](./src/loss-functions/image-20210726203705056.png)

文章首先提出了一种对于Region Propasl（RP）和Ground Truth（GT）之间的新评估方式，即CDIoU。可以发现，它虽然没有直接中心点距离和长宽比，但最终的计算结果是有反应出RP和GT的差异。计算公式如下：
$$
diou = \frac{||RP-GT||_2}{4MBR's diagonal}\\
=\frac{AE+BF+CG+DH}{4WY}
$$

$$
CDIoU = IoU + \lambda(1 - diou)
$$

对比以往直接计算中心点距离或是形状相似性的损失函数，CDIoU能更合理地评估RP和GT的差异并且有效地降低计算成本。然后，根据上述的公式，CDIoU Loss可以定义为：
$$
L_{CDIoU} = L_{IoU_s} + diou
$$
通过观察这个公式，可以直观地感受到，在权重迭代过程中，模型不断地将RP的四个顶点拉向GT的四个顶点，直到它们重叠为止，如下图所示：

![image-20210726204000099](./src/loss-functions/image-20210726204000099.png)

## 分类损失（Classification Loss）

### Entropy

### Cross Entropy

### K-L Divergence

### Dice Loss

### Focal Loss

### Tversky loss

