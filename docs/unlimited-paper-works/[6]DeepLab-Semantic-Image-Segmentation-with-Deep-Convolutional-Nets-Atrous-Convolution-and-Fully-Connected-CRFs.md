# DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

### 这篇笔记的写作者是[VisualDust](https://github.com/visualDust)。

原论文：[Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915.pdf)。

在之前的语义分割网络中，分割结果往往比较粗糙，原因主要有两个，一是因为池化导致丢失信息，二是没有利用标签之间的概率关系，针对这两点，作者提出了针对性的改进。首先使用**空洞卷积（Atrous Convolution）**，避免池化带来的信息损失，然后使用**条件随机场（CRF）**，进一步优化分割精度。阅读这篇论文应关注的重点问题是：

- 空洞卷积
- 条件随机场

> In this work we address the task of semantic image segmentation with Deep Learning and make three main contributions that are experimentally shown to have substantial practical merit. First, we highlight convolution with upsampled filters, or 'atrous convolution', as a powerful tool in dense prediction tasks. Atrous convolution allows us to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks. It also allows us to effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation. Second, we propose atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales. ASPP probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales. Third, we improve the localization of object boundaries by combining methods from DCNNs and probabilistic graphical models. The commonly deployed combination of max-pooling and downsampling in DCNNs achieves invariance but has a toll on localization accuracy. We overcome this by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance. Our proposed "DeepLab" system sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 79.7% mIOU in the test set, and advances the results on three other datasets: PASCAL-Context, PASCAL-Person-Part, and Cityscapes. All of our code is made publicly available online.

---

空洞卷积的主要作用是在增大感受野的同时，不增加参数数量，而且VGG中提出的多个小卷积核代替大卷积核的方法，只能使感受野线性增长，而多个空洞卷积串联，可以实现指数增长。

关于条件随机长，简单来讲就是每个像素点作为节点，像素与像素间的关系作为边，即构成了一个条件随机场。通过二元势函数描述像素点与像素点之间的关系，鼓励相似像素分配相同的标签，而相差较大的像素分配不同标签，而这个“距离”的定义与颜色值和实际相对距离有关。所以这样CRF能够使图片在分割的边界出取得比较好的效果。

咳咳，还没有开始写
