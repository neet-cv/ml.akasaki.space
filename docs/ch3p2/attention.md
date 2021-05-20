# 注意力机制

注意力机制在很多AI领域内得到了成功的应用。这是人工神经网络在模仿人类进行决策过程的重要发展。

> In humans, Attention is a core property of all perceptual and cognitive operations. Given our limited ability to process competing sources, attention mechanisms select, modulate, and focus on the information most relevant to behavior.

这段文字摘自Alana de Santana Correia, and Esther Luna Colombini的论文 [ATTENTION, PLEASE ! A SURVEY OF NEURAL ATTENTION MODELS IN DEEP LEARNING](https://arxiv.org/abs/2103.16775)。

## 介绍

你应该注意到了，在你的视野中，只有一部分区域是很清晰的。对于视野周围的场景，你往往需要转转眼珠，把视野朝向它，才能完全看清。或者，你还发现，比起历史老师开始强调重点，你似乎对下课铃声的响起更加敏感——这就是注意力。你所处的环境包含着远超你的处理能力的信息，而注意力机制让你的大脑集中精力处理你视野中心的场景，或是你“更应该”关心的事物。

在这一节，我们对基于神经网络的计算机视觉任务中如何使用注意力机制进行简单的了解。在这一节中，我们主要了解如何产生可以学习的注意力以及如何实现这种注意力。

注意力是一个非常模糊的概念。如果你想把它应用于神经网络中，你需要清晰地表达注意力，无论是用公式还是用什么其他的。直接一点说，注意力应该也是训练得到的参数，就和卷积的参数或是全连接的参数一样。这些参数能让网络开始学习“滤除”图像的不相关部分，以便对该任务做出更清晰，更可靠的判断。使用注意力机制能让你的神经网络兼容性更强，更具鲁棒性。