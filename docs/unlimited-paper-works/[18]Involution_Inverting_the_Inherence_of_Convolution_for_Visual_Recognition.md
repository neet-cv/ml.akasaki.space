# Involution: Inverting the Inherence of Convolution for Visual Recognition

### 这篇笔记的写作者是[AsTheStarsFall](https://github.com/asthestarsfalll)。

> 论文名称：[*Involution: Inverting the Inherence of Convolution for Visual Recognition*](https://arxiv.org/abs/2103.06255)
>
> 作者：Duo Li， Jie Hu， Changhu Wang， Xiangtai Li， Qi She， Lei Zhu， Tong Zhang， Qifeng Chen， The Hong Kong University of Science and Technology， ByteDance AI Lab， Peking University， Beijing University of Posts and Telecommunications

# Convolution

1. [空间无关性(spatial agnostic)](https://arxiv.org/pdf/1805.12177.pdf)：same kernel for different position
   - 优点：参数共享，平移等变
   - 缺点：不能灵活改变参数，卷积核尺寸不能过大，只能通过堆叠来扩大感受野、捕捉长距离关系
2. 通道特异性(channel specific)：different kernels for different channels
   - 优点：充分提取不同通道上的信息
   - 缺点：有冗余

Convolution kernel 尺寸为 B,C_out,C_in,K,K

# Involution

与convolution不同，involution拥有**完全相反**的性质：

1. 空间特异性：kernel privatized for different position
2. 通道不变性：kernel shared across different channels

involution kernel 的尺寸为B,G,KK,H,W.

# how to generate involution kernels

kernel generated based on input featrue map(self-attention的一种体现？) to ensure kernel size aligned with the input tensor size

一种简单的kernel generation function,方便起见以一个像素为例

![image-20210426192156487](https://gitee.com/Thedeadleaf/images/raw/master/20210507124123.png)

1. inputs为维度1×1×C；
2. 线性变换：$W_0$：通道压缩，节省计算量；$W_1$：首先变为1×1×(K×K×G)，再拆分为G组，最后变换为K×K×G；（其生成的卷积核包含了所有通道上的信息，对不同通道之间的信息交换有一定的作用）
3. 生成的kernel与(i,j)像素领域进行乘加操作，因为维度不同，需要进行广播，得到大小为k×k×C；
4. 最后进行聚合，输出大小为1×1×C。

```python
class Involution(nn.Module):
    def __init__(self, channel, group, kernel, s):
        super(Involution, self).__init__()
        self.channel = channel
        self.group = group
        self.kernel_size = kernel
        ratio=4

        self.o = nn.AvgPool2d(s, s) if s > 1 else nn.Identity()
        self.reduce = nn.Sequential(
            nn.Conv2d(channel, channel//ratio, 1),
            nn.BatchNorm2d(channel//ratio),
            nn.ReLU(inplace=True)
        )
        self.span = nn.Conv2d(channel//ratio, kernel**2*group, 1)
        # 从一个Batch中提取出卷积滑动的局部区域块，较难理解，建议自行百度
        # 普通的卷积操作实际上就相当于将feature map unfold与conv kernel乘加之后再fold
        self.unfold = nn.Unfold(
            kernel_size=kernel, padding=(kernel-1)//2, stride=s)

    def forward(self, x):
        kernel = self.span(self.reduce(self.o(x)))  # B,KKG,H,W
        B, _, H, W = kernel.shape
        kernel = kernel.view(B, self.group, self.kernel_size **
                             2, H, W).unsqueeze(2)  # B,G,1,kk,H,W，unsqueeze：增加一个维度用于广播

        x_unfolded = self.unfold(x)  # B,CKK,HW
        x_unfolded = x_unfolded.view(
            B, self.group, self.channel//self.group, self.kernel_size**2, H, W)# B,G,C/G,KK,H,W

        out = (kernel*x_unfolded).sum(dim=3)  # B,G,C/G,H,W
        out = out.view(B, self.channel, H, W) # B,C,H,w
        return out

```

更多：

1. 对Involution kernel的生成方式进行更多的探索；
2. 进一步探索Convolution-Involution的混合结构。

# Involution	vs	Convolution

优点：

1. 参数量和计算量都很少

   - 对于Convolution，其参数量为：
     $$
     K^2C_{in}C_{out}
     $$
     计算量大约为：
     $$
     HWK^2C_{in}C_{out}
     $$

   - 对于Involution，其参数量为：
     $$
     \frac{C^2+CGK^2}{r}
     $$
     计算量大约为：
     $$
     HWK^2C
     $$

   可以看到，involution的计算量与通道数呈线性关系。

2. 能有效建模长距离关系

   相较于Convolution，involution kernel可以使用更大的卷积核而不过多增加其参数量，其感受野也就越大。

3. involution是动态的，而convolution是静态的。

缺点：

1. 通道间的信息交换在一定程度上受到影响

   虽然同一组内共享同一个kernel，但是不同组通道间的信息交换还是会受到影响。

2. 速度相较于Convolution没有优势



# Relation to Self-Attention

> self-attention可以看作广义involution的一种实例

可以看到与self-attention之间的相似性：

![image-20210427212347084](https://gitee.com/Thedeadleaf/images/raw/master/20210507124139.png)

相似：

1. 其中H可以类比为involution中的G；
2. self-attention中每个位置的关系矩阵可以类比为involution中每个位置的kernel。

不同：

1. 相比于self-attention，Involution潜在编码了位置信息，而self-attention需要position encoding来区分位置信息.
2. 不在需要使用Q-K，仅依靠单个像素生成kernel，而非依靠像素间的关系生成attention map.

总结：self-attention是Involution的一种实例化，且Involution的表达更为宽泛和简洁。

# Ablantion    Analysis

![image-20210427213135224](https://gitee.com/Thedeadleaf/images/raw/master/20210507124142.png)

可以看到：

- Involution能在不显著提升参数量和计算量的前提下，增大kernel的感受野，提升网络性能;
- 在显著降低计算量和参数量的情况下，准确度损失却不大。

# 其他

1. 关于卷积的可替代性

   特征在空间位置上差异明显，我们更需要注意长距离关系时，involution或许是个好的选择。

2. 训练与优化

   不同于convolution，involution实际上是二阶优化，需要优化的并不是kernel，而是kernel生成函数里的参数，这就会造成很多问题（最近的transformer优化过程也有很多问题），作者建议对于某些网络需要使用gradient clipping等方法来进行更好的优化。

3. 硬件支持

   involution的优化并没有convolution好，也没有相应硬件的支持，因此虽然参数量和计算量都减小了，但是实际并没有convolution快，作者建议使用CUDA编写Involution。