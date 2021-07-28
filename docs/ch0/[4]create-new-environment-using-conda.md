# 使用conda创建一个环境

---

## 为什么

在以后的时光里，你会接触到很多不同的深度学习方法和子领域。但是只要你还在用python，你就脱离不了大量的包和依赖。并且，在不同的深度学习项目中，往往要使用不同包甚至相同包的不同版本。你肯定不想每次创建或切换项目都重新打点一次你的包版本和依赖。

好消息是你可以使用工具来管理你的环境，也就是说，你可以在一台电脑上创建多个人不同的环境，就像是虚拟机，需要用哪个就激活哪个。这样能够大大节省不必要浪费掉的时间，让你全心全意投入学习和工作。

---

## 安装anaconda

anaconda是可靠的知名虚拟环境管理工具。你可以在[anaconda的官网](https://www.anaconda.com/)找到和下载安装包，并根据提示进行安装。

- 提示：在windows和mac上，你可以直接下载应用程序安装程序或镜像文件；在linux上，你可以更方便的使用包管理器来完成anaconda的安装。当然，anaconda也是有替代品的，比如更小巧的miniconda。

---

## 切换源

就像linux的包管理器一样，anaconda不但是个虚拟环境管理器，也是个包管理器。源可以理解为包管理器下载包的远程服务器。源有很多个，就像你下载软件时可能会有国内的下载站或是国外的下载站。由于中国大陆地区网络的特殊性，这里建议你**切换到国内源来提高包的下载速度**。

国内较为出名的公有源有清华源和中科大源。你可以通过下面的命令行将它们添加到你的conda源：

```bash
# 添加清华源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge 
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
```

```bash
# 添加中科大源
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/cloud/menpo/
```

你还可以通过下面这行命令让conda在搜索到包时显示它是在哪个源找到的：

```bash
conda config --set show_channel_urls yes
```

出了通过命令直接修改这些信息，你也可以修改anaconda的配置文件。详情可以自行探索。

---

## 创建环境

创建一个名称为mxnet的环境，并指定这个环境的python版本为3.8

```bash
conda create -n mxnet python=3.8
```

---

## 安装Tensorflow

咳咳，百度。

## 其他

### 有conda为什么要使用pip？

conda除了是一个虚拟环境管理器之外，也是一个包管理器。也就是说，你可以通过conda安装这些包。但是为什么上面使用了pip？放心，conda和pip并不会冲突。它们可以同时存在。实际上，有的时候通过pip安装和通过conda安装是等效的。当然也有的时候是不等效的。不过这在大多数情况下影响不大。总之目前你可以认为通过它们两个安装某个包是一样的，有的区别你会在以后的时间里慢慢明白。对我而言，我觉得单纯使用conda作为虚拟环境管理并使用pip安装一些包体验十分良好。并且，mxnet官网提供的安装指南使用了pip。

### pip切换源

和conda同理，**将pip切换至国内源可以提高包的下载速度**。你可以从下面的命令行中选择一条来执行。

```bash
# 阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 腾讯源
pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
# 豆瓣源
pip config set global.index-url http://pypi.douban.com/simple/
```

在这里推荐截止到本文写作日期时表现更加稳定的阿里源。

