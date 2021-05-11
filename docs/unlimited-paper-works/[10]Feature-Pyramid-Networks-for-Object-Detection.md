# Feature Pyramid Networks for Object Detection

### 这篇笔记的写作者是[VisualDust](https://github.com/visualDust)。

这篇论文就是大家熟知的FPN了。FPN是比较早期的一份工作（CVPR2017），在当时具有很多亮点。该论文提出，特征金字塔是识别系统中用于检测不同比例物体的基本组件（号称手工特征设计时代的万金油），比如在OpenCV库的特征匹配Cascade分类器用于人脸识别中使用特征金字塔模型+AdaBoost提取不同尺度特征经行分类等。

不过后来很多基于深度学习的工作都不太用pyramid representations了，因为这种设计比较消耗内存和计算资源。

> Feature pyramids are a basic component in recognition systems for detecting objects at different scales. But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive. In this paper, we exploit the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. A top-down architecture with lateral connections is developed for building high-level semantic feature maps at all scales. This architecture, called a Feature Pyramid Network (FPN), shows significant improvement as a generic feature extractor in several applications. Using FPN in a basic Faster R-CNN system, our method achieves state-of-the-art single-model results on the COCO detection benchmark without bells and whistles, surpassing all existing single-model entries including those from the COCO 2016 challenge winners. In addition, our method can run at 5 FPS on a GPU and thus is a practical and accurate solution to multi-scale object detection. Code will be made publicly available.

咳咳，还没有开始写，占坑