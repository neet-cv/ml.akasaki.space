---
title: 凸优化学习
date: 2022-03-12 15:19:59
author: Breeze Shane
top: false
toc: true
mathjax: true
categories: 
 - AritficialIntelligence
 - Mathematics
 - Convex
tags: 
 - AritficialIntelligence
 - Mathematics
 - Convex
---

<details>

<summary>参考</summary>

- **Main** [凸函数 - 文再文讲义](https://bicmr.pku.edu.cn/~wenzw/optbook/lect/03_functions_newhyx.pdf)
- [为什么矩阵内积的定义包含一个迹运算？](https://www.zhihu.com/question/274052744)
- [矩阵的 Frobenius 范数与 trace （迹）的关系及其求偏导法则](https://blog.csdn.net/frx_bwcx/article/details/108194176?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-3-108194176.pc_agg_new_rank&utm_term=%E7%9F%A9%E9%98%B5%E5%86%85%E7%A7%AF%E4%B8%8E%E8%BF%B9%E7%9A%84%E5%85%B3%E7%B3%BB&spm=1000.2123.3001.4430)
- [线性代数笔记 Frobenius 范数和矩阵的迹之间的关系](https://blog.csdn.net/qq_40206371/article/details/121123061?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_ecpm_v1~rank_v31_ecpm-1-121123061.pc_agg_new_rank&utm_term=%E7%9F%A9%E9%98%B5%E5%86%85%E7%A7%AF%E4%B8%8E%E8%BF%B9%E7%9A%84%E5%85%B3%E7%B3%BB&spm=1000.2123.3001.4430)
- [利用分塊矩陣證明 det(AB)=(det A)(det B)](https://ccjou.wordpress.com/2009/05/08/%E5%88%A9%E7%94%A8%E5%88%86%E5%A1%8A%E7%9F%A9%E9%99%A3%E8%AD%89%E6%98%8E-detabdet-adet-b/)
- [为什么对数函数的泰勒展开式要用ln（x+1）而不用lnx？](https://www.zhihu.com/question/415271590)
- [墨卡托级数](https://zh.wikipedia.org/wiki/%E5%A2%A8%E5%8D%A1%E6%89%98%E7%B4%9A%E6%95%B8)
- [Open set](https://en.wikipedia.org/wiki/Open_set)
- [Closed set](https://en.wikipedia.org/wiki/Closed_set)
- [方向导数](https://zh.wikipedia.org/wiki/%E6%96%B9%E5%90%91%E5%AF%BC%E6%95%B0)


</details>

## 基础知识

### 梯度(Gradient)

:::info 定义

给定函数$f : \mathbb{R}^n \to \mathbb{R}$,且$f$在点$x$的一个邻域内有意义,若存在向量$g \in \mathbb{R}^n$满足:

$$
\lim_{p\to 0}\frac{f(x+p) - f(x) - g^\top p}{||p||} = 0,
$$

其中$||\cdot||$是任意的向量范数,就称$f$在点$x$处可微(或Fréchet可微).此时$g$称为$f$在点$x$处的梯度,记作$\nabla f(x)$.如果对区域$D$上的每一个点$x$都有$\nabla f(x)$存在,则称$f$在$D$上可微.

:::

若$f$在点$x$处的梯度存在,在定义式中令$p=\epsilon e_i$,$e_i$是第$i$个分量为1的单位向量,可知$\nabla f(x)$的第$i$个分量为$\frac{\partial f(x)}{\partial x_i}$.因此,

$$
\nabla f(x) = \Big[ \frac{\partial f(x)}{\partial x_1}, \frac{\partial f(x)}{\partial x_2}, \cdots, \frac{\partial f(x)}{\partial x_n} \Big]^\top
$$

### 海瑟矩阵(Hessian Matrix)

:::info 定义

如果函数$f(x):\mathbb{R}^n \to \mathbb{R}$在点$x$处的二阶偏导数$\frac{\partial^2f(x)}{\partial x_i \partial x_j} \; i, j = 1, 2, \cdots , n$都存在,则

$$
\nabla^2f(x) = 
\begin{bmatrix}
\frac{\partial^2f(x)}{\partial x_1^2} & \frac{\partial^2f(x)}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2f(x)}{\partial x_1 \partial x_n} \\

\frac{\partial^2f(x)}{\partial x_2 \partial x_1} & \frac{\partial^2f(x)}{\partial x_2^2} & \cdots & \frac{\partial^2f(x)}{\partial x_2 \partial x_n} \\

\vdots & \vdots & \ddots & \vdots \\

\frac{\partial^2f(x)}{\partial x_n \partial x_1} & \frac{\partial^2f(x)}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2f(x)}{\partial x_n^2}
\end{bmatrix}
$$

称为$f$在点$x$处的海瑟矩阵.

:::

当$\nabla^2f(x)$在区域$D$上的每个点$x$处都存在时,称$f$在$D$上二阶可微.若$\nabla^2f(x)$在$D$上还连续,则称$f$在$D$上二阶连续可微,可以证明此时海瑟矩阵是一个对称矩阵.

### 矩阵变量函数的导数

多元函数梯度的定义可以推广到变量是矩阵的情形.对于以$m\times n$矩阵$X$为自变量的函数$f(X)$,若存在矩阵$G \in \mathbb{R}^{m\times n}$满足

$$
\lim_{V\to 0}\frac{f(X+V)-f(X)-\langle G, V\rangle}{||V||} = 0,
$$

其中$||\cdot||$是任意矩阵范数,就称矩阵变量函数$f$在$X$处Fréchet可微,
称$G$为$f$在Fréchet可微意义下的梯度.类似于向量情形,矩阵变量函数$f(X)$的梯度可以用其偏导数表示为

$$
\nabla f(x) = 
\begin{bmatrix}
\frac{\partial f(x)}{\partial x_{11}} & \frac{\partial f(x)}{\partial x_{12}} & \cdots & \frac{\partial f(x)}{\partial x_{1n}} \\

\frac{\partial f(x)}{\partial x_{21}} & \frac{\partial f(x)}{\partial x_{22} } & \cdots & \frac{\partial f(x)}{\partial x_{2n}} \\

\vdots & \vdots & \ddots & \vdots \\

\frac{\partial^2f(x)}{\partial x_{m1}} & \frac{\partial f(x)}{\partial x_{m2}} & \cdots & \frac{\partial f(x)}{\partial x_{mn}}
\end{bmatrix}
$$

其中$\frac{\partial f(x)}{\partial x_{ij}}$表示$f$关于$x_{ij}$的偏导数.

在实际应用中,矩阵Fréchet可微的定义和使用往往比较繁琐,为此我们需要介绍另一种定义——Gâteaux可微.

:::info Gâteaux可微定义

设$f (X)$为矩阵变量函数,如果对任意方向$V \in \mathbb{R}^{m\times n}$,存在矩阵$G \in \mathbb{R}^{m\times n}$满足

$$
\lim_{t \to 0} \frac {f(X+tV) - f(X) - t \langle G, V \rangle}{t} = 0,
$$

则称$f$关于$X$是Gâteaux可微的.满足上式的$G$称为$f$在$X$处在Gâteaux
可微意义下的梯度.

:::

可以证明,当$f$是Fréchet可微函数时,$f$也是Gâteaux可微的,且这两种意义下的梯度相等.

下面是三种经典的例子

<details>

<summary>线性函数</summary>


$f(X) = tr(AX^\top B)$,其中$A \in \mathbb{R}^{p\times n}, B \in \mathbb{R}^{m\times p},X \in \mathbb{R}^{m\times n}$对任意方向$V \in \mathbb{R}^{m\times n}$以及$t\in \mathbb{R}$,有

$$
\lim_{t\to 0}\frac{f(X+tV)-f(X)}{t} = \lim_{t\to 0} \frac{tr(A(X+tV)^\top B) - tr(AX^\top B)}{t} = tr(AV^\top B) = tr(V^\top BA) = \langle BA, V\rangle.
$$

因此,$\nabla f(x) = BA$.

</details>

<details>

<summary>二次函数</summary>

$f(X,Y) = \frac{1}{2} || XY - A ||_F^2$,其中$(X,Y)\in \mathbb{R}^{m\times p}\times \mathbb{R}^{p\times n}$对变量$Y$,取任意方向$V$以及充分小的$t\in \mathbb{R}$,有

$$
\begin{aligned}
f(X,Y+tV) - f(X,Y) 
&= \frac{1}{2}||X(Y+tV)-A||_F^2-\frac{1}{2}||XY-A||_F^2 \\
&= \frac{1}{2}tr\Big\{ [X(Y+tV)-A]^\top [X(Y+tV)-A] \Big\} - \frac{1}{2}tr\Big[ (XY-A)^\top(XY-A) \Big] \\
&= \frac{1}{2}tr\Big[(XY+tXV-A)^\top (XY+tXV-A) \Big] - \frac{1}{2}tr\Big[ (XY-A)^\top(XY-A) \Big] \\
&= \frac{1}{2}tr\Big[(XY+tXV-A)^\top (XY-A) \Big] + \frac{1}{2}tr\Big[(XY+tXV-A)^\top (tXV) \Big] - \frac{1}{2}tr\Big[ (XY-A)^\top(XY-A) \Big] \\
&= \frac{1}{2}tr\Big[(XY-A)^\top(XY+tXV-A) \Big] + \frac{1}{2}tr\Big[(tXV)^\top (XY+tXV-A) \Big] - \frac{1}{2}tr\Big[ (XY-A)^\top(XY-A) \Big] \\
&= \frac{1}{2}tr\Big[(XY-A)^\top(XY-A) \Big] + \frac{1}{2}tr\Big[(XY-A)^\top(tXV) \Big] + \frac{1}{2}tr\Big[(tXV)^\top (XY+tXV-A) \Big] - \frac{1}{2}tr\Big[ (XY-A)^\top(XY-A) \Big] \\
&= \frac{1}{2}tr\Big[(XY-A)^\top(tXV) \Big] + \frac{1}{2}tr\Big[(tXV)^\top (XY-A) \Big] + \frac{1}{2}tr\Big[(tXV)^\top (tXV) \Big]  \\
&= t\cdot tr\Big[(XV)^\top(XY-A) \Big] + \frac{t^2}{2}tr\Big[(XV)^\top (XV) \Big]  \\
&= t\cdot tr\Big[V^\top \cdot X^\top(XY-A) \Big] + \frac{t^2}{2}||XV||_F^2  \\
&=t\langle V,X^\top(XY-A) \rangle + \mathcal{O}(t^2)
\end{aligned}
$$

由定义知$\frac{\partial f}{\partial Y} = X^\top(XY-A)$.

对变量$X$,同理可得$\frac{\partial f}{\partial X} = (XY-A)Y^\top$.

</details>

<details>

<summary>ln-det函数</summary>

$f(X) = \ln(\det(X)), \; X\in \mathcal{S}_{++}^n$,给定义$X \succ 0$,对任意方向$V \in \mathcal{S}^n$以及$t\in \mathbb{R}$,我们有

$$
\begin{aligned}
f(X+tV) - f(X)
&= \ln(\det(X+tV)) - \ln(\det(X)) \\
&= \ln(\det(X^{\frac{1}{2}}(I + tX^{-\frac{1}{2}}VX^{-\frac{1}{2}}) X^{\frac{1}{2}} )) - \ln(\det(X)) \\
&= \ln\Big( \frac{\det[ X^{\frac{1}{2}}(I + tX^{-\frac{1}{2}}VX^{-\frac{1}{2}}) X^{\frac{1}{2}} ]}{\det(X)} \Big) \\
&= \ln\Big( \frac{\det (X^{\frac{1}{2}})\det(I + tX^{-\frac{1}{2}}VX^{-\frac{1}{2}}) \det(X^{\frac{1}{2}}) }{\det(X)} \Big) \\
&= \ln\Big( \frac{\det (X)\det(I + tX^{-\frac{1}{2}}VX^{-\frac{1}{2}}) }{\det(X)} \Big) \\
&= \ln(\det(I + tX^{-\frac{1}{2}}VX^{-\frac{1}{2}})) \\
\end{aligned}
$$

由于$X^{-\frac{1}{2}}VX^{-\frac{1}{2}}$是对称矩阵,所以他可以正交对角化,不妨设其特征值为$\lambda_1,\lambda_2,\dots,\lambda_n$,则

$$
\begin{aligned}
\ln(\det(I + tX^{-\frac{1}{2}}VX^{-\frac{1}{2}})) 
&= \ln\prod_{i=1}^{n}(1+t\lambda_i) \\
&= \sum_{i=1}^{n}\ln(1+t\lambda_i) \\
&= \sum_{i=1}^{n}t\lambda_i + \mathcal{O}(t^2) \\
&= t \Big\langle (X^{-1})^\top, V \Big\rangle + \mathcal{O}
\end{aligned}
$$

因此,我们可以根据Gâteaux可微定义得到结论$\nabla f(X)=(X^{-1})^\top$.

注意,$\sum_{i=1}^{n}\ln(1+t\lambda_i) = \sum_{i=1}^{n}t\lambda_i + \mathcal{O}(t^2)$这里其实是应用了级数来计算的.

$$
\because \ln z = \sum_{i=1}^{n}(-1)^{i+1}\cdot\frac{1}{i}(z-1)^i + \mathcal{O}((z-1)^2) \\
\therefore \ln(1+x)=\sum_{i=1}^{n}\frac{(-1)^{i+1}}{i}x^i + \mathcal{O}(x^2)
$$

而上面那一步是用$\ln (1+x) = x + \mathcal{O}(x^2)$来作和直接代换的.

</details>

:::tip 计算中用到的定理及结论

$$
tr(A^\top B) = \langle A,B \rangle = \sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}b_{ij}
$$

易知

$$
tr(A^\top A) = \langle A,A \rangle = \sum_{i=1}^{n}\sum_{j=1}^{n}a_{ij}^2 = ||A||_F^2
$$

$\ln(1+x)$的泰勒展开式:

$$
\ln(1+x)=\sum_{i=1}^{n}\frac{(-1)^{i+1}}{i}x^i + \mathcal{O}(x^2)
$$

:::

<details>

<summary>ln(1+x)级数的证明</summary>

已知当$|x| < 1$时,$\frac{1}{1+x} = \frac{1}{1-(-x)}=\sum_{n=0}^{\infty}(-x)^n$

而对于$\ln(1+x)$求导得$\frac{1}{1+x}$,所以我们有:

$$
\begin{aligned}
\ln(1+x) 
&= \int\frac{1}{1+x}dx 
= \int\frac{1}{1-(-x)}dx \\
&= \int\sum_{n=0}^{\infty}(-x)^ndx 
= \sum_{n=0}^{\infty}\int(-x)^ndx \\
&= \sum_{n=0}^{\infty}-\frac{1}{n+1}(-x)^{n+1} \\
&= \sum_{n=1}^{\infty}-\frac{1}{n}(-x)^{n}
\end{aligned}
$$

我查阅了Wiki的资料,证明的思路不一样,也值得一看,下面贴出Wiki给的证明:

在Wiki上给出了这个级数的名字:墨卡托级数(或称牛顿-墨卡托级数).

我们有等比数列$\{(-t)^{n-1}\}$,可以得出:

$$
\begin{aligned}
&\qquad 1-t+t^{2}-\cdots +(-t)^{n-1}=\frac{1-(-t)^{n}}{1+t} \\
&\Leftrightarrow \frac {1}{1+t}=1-t+t^{2}-\cdots +(-t)^{n-1}+{\frac {(-t)^{n}}{1+t}} \\
&\Leftrightarrow \int _{0}^{x}{\frac {dt}{1+t}}=\int _{0}^{x}\left(1-t+t^{2}-\cdots +(-t)^{n-1}+{\frac {(-t)^{n}}{1+t}}\right)\,dt \\
&\Leftrightarrow \displaystyle \ln(1+x)=x-{\frac {x^{2}}{2}}+{\frac {x^{3}}{3}}-\cdots +(-1)^{n-1}{\frac {x^{n}}{n}}+(-1)^{n}\int _{0}^{x}{\frac {t^{n}}{1+t}}\,dt
\end{aligned}
$$

当$-1 < x \leq 1$时余项$(-1)^{n}\int _{0}^{x}{\frac {t^{n}}{1+t}}\,dt$会在$n \to \infty$时趋近于$0$.

</details>

### 广义实值函数

:::info 定义

令$\overline{\mathbb{R}} := \mathbb{R} \cup \{\pm\infty\}$为广义实数空间,则映射$f:\mathbb{R}^n\to \overline{\mathbb{R}}$称为广义实值函数.

和数学分析一样,我们规定
$$
-\infty < a < +\infty, \; \forall a \in \mathbb{R}, \\
(+\infty) + (+\infty) = +\infty, \\
(+\infty) + a = +\infty, \; \forall a \in \mathbb{R}
$$

:::

### 适当函数

:::info 定义

给定广义实值函数$f$和非空集合$\mathcal{X}$.如果存在$x \in \mathcal{X}$使得$f(x) < +\infty$,并且对任意的$x \in \mathcal{X}$,都有$f(x)>−\infty$,那么称函数$f$关于集合$\mathcal{X}$ 是适当的.

:::

概括来说,适当函数$f$的特点是“至少有一处取值不为正无穷”,以及“处处取值不为负无穷”.

### 下水平集

:::info 定义

对于广义实值函数$f : \mathbb{R}^n → \overline{\mathbb{R}}$,

$$
C_\alpha = \{x | f (x) \leq α \}
$$
称为$f$的$\alpha$-下水平集.

:::

### 上方图

:::info 定义

对于广义实值函数$f : \mathbb{R}^n → \overline{\mathbb{R}}$,

$$
epi\; f= \{ (x, t) ∈ \mathbb{R}^{n+1} |f (x) \leq t\}
$$

称为$f$的上方图.

:::

### 闭函数

:::info 定义

设$f : \mathbb{R}^n → \overline{\mathbb{R}}$为广义实值函数,若$epi \;f$为闭集,则称$f$为闭函数.

:::

<details>

<summary>开集,闭集的概念</summary>

Wiki中文给出的定义看得有点迷,咱们还是看看英文的定义吧.

> **Open set** : A subset $U$ of the Euclidean n-space $R^n$ is open if, for every point $x$ in $U$, there exists a positive real number $\varepsilon$ (depending on $x$) such that a point in $R^n$ belongs to $U$ as soon as its Euclidean distance from $x$ is smaller than $\varepsilon$. Equivalently, a subset $U$ of $R^n$ is open if every point in $U$ is the center of an open ball contained in $U$.

> **Closed set** : By definition, a subset $A$ of a topological space $(X, \tau)$ is called closed if its complement $X\setminus A$ is an open subset of $(X, \tau)$; that is, if $X\setminus A\in \tau$. $X\setminus A\in \tau$. A set is closed in $X$ if and only if it is equal to its closure in $X$. Equivalently, a set is closed if and only if it contains all of its limit points. Yet another equivalent definition is that a set is closed if and only if it contains all of its boundary points. Every subset $A\subseteq X$ is always contained in its (topological) closure in $X$, which is denoted by $\operatorname{cl}_{X}A$; that is, if $A\subseteq X$ then $A\subseteq \operatorname {cl} _{X}A$. Moreover, $A$ is a closed subset of $X$ if and only if $A=\operatorname {cl} _{X}A$.

</details>

### 下半连续函数

:::info 定义

设广义实值函数$f : \mathbb{R}^n → \overline{\mathbb{R}}$,若对$\forall x\in \mathbb{R}^n$,有

$$
\lim_{y\to x}\inf{f(y)} \geq f(x),
$$

则称$f(x)$为下半连续函数.

:::

:::tip 上半连续函数

类似地,我们可以这样定义上半连续函数:

设广义实值函数$f : \mathbb{R}^n → \overline{\mathbb{R}}$,若对$\forall x\in \mathbb{R}^n$,有

$$
\lim_{y\to x}\sup{f(y)} \leq f(x),
$$

则称$f(x)$为上半连续函数.

:::

### 闭函数和下半连续函数的关系

虽然表面上看这两种函数的定义方式截然不同,但闭函数和下半连续函数是等价的.

:::info 定理

设广义实值函数$f : \mathbb{R}^n → \overline{\mathbb{R}}$,则以下命题等价:

1. $f(x)$的任意$\alpha$-下水平集都是闭集;
2. $f(x)$是下半连续的;
3. $f(x)$是闭函数.

:::

闭(下半连续)函数间的简单运算会保持原有性质:
1. **加法**: 若$f$与$g$均为适当的闭(下半连续)函数,并且$\mathrm{dom}\,f \cap \mathrm{dom}\,g \neq \varnothing$,则$f + g$也是闭(下半连续)函数.其中适当函数的条件是为了避免出现未定式$(−\infty)+(+\infty)$的情况;
2. **仿射映射的复合**: 若$f$为闭(下半连续)函数, 则$f(Ax + b)$也为闭(下半连续)函数;
3. **取上确界**: 若每一个函数$f_α$均为闭(下半连续)函数,则$\sup_\alpha f_\alpha(x)$也为闭(下半连续)函数.

## 凸函数

:::info 定义

$f:\mathbb{R}^n \to \mathbb{R}$为适当函数,如果$\mathrm{dom}\,f$是凸集,且
$$
f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)
$$
对所有$x,y \in \mathrm{dom}\,f,\;0 \leq \theta \leq 1$都成立,则称$f$是凸函数.

:::

若$f$是凸函数,则$-f$是凹函数.

若对所有$x,y \in \mathrm{dom}\,f,\,x\neq y,\;0 < \theta < 1$,有
$$
f(\theta x + (1-\theta)y) < \theta f(x) + (1-\theta)f(y)
$$
则称$f$是严格凸函数.

### 一元凸函数和一元凹函数的例子

凸函数:
1. 仿射函数: $\forall a, b \in \mathbb{R}$, $ax+b$是$\mathbb{R}$上的凸函数.
2. 指数函数: $\forall a \in \mathbb{R}$,$e^{ax}$是$\mathbb{R}$上的凸函数.
3. 幂函数: $\forall \alpha \in (-\infty, 0] \cup [1, +\infty)$, $x^\alpha$是$\mathbb{R}_{++}$上的凸函数.
4. 绝对值的幂: $\forall p \geq 1$,$|x|^p$是$\mathbb{R}$上的凸函数.
5. 负熵: $x\log x$是$\mathbb{R}_{++}$上的凸函数.

凹函数:
1. 仿射函数:对任意$a, b \in \mathbb{R}$, $ax+b$是$\mathbb{R}$上的凹函数.
2. 幂函数:$\forall \alpha \in [0, 1]$, $x^\alpha$是$\mathbb{R}_{++}$上的凹函数.
3. 对数函数:$\log x$是$\mathbb{R}_{++}$上的凹函数.

可以看出,所有的仿射函数既是凸函数,又是凹函数.所有的范数都是凸函数.这两个结论对多元函数同样成立.

### 多元凸函数的例子

**欧氏空间**$\mathbb{R}^n$**中的例子**:

仿射函数: $f(x) = a^\top x + b$

范数: $||x||_p = (\sum_{i=1}^{n}|x_i|^p)^{\frac{1}{p}} \; (p \geq 1)$;特别地, $||x||_\infty = \max_k |x_k|$

**矩阵空间**$\mathbb{R}^{m\times n}$**中的例子**:

仿射函数:

$$
f(X) = tr(A^\top X) + b = \sum_{i=1}^m\sum_{j=1}^nA_{ij}X_{ij} + b
$$

谱范数:

$$
f(X) = ||X||_2 = \sigma_{\max}(X) = (\lambda_{\max}(X^\top X))^{\frac{1}{2}}
$$

### 强凸函数

:::info 定义1

若存在常数$m > 0$,使得

$$
g(x) = f(x) - \frac{m}{2}||x||^2
$$

为凸函数,则称$f(x)$为强凸函数,其中$m$为强凸参数.为了方便,我们也称$f(x)$为$m$-强凸函数.

:::

:::info 定义2

若存在常数$m > 0$,使得 $\forall x, y \in \mathrm{dom} \,f$以及$\theta \in (0,1)$,有

$$
f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y) - \frac{m}{2}\theta(1-\theta)||x - y||^2
$$

则称$f(x)$为强凸函数,其中$m$为强凸参数.

:::

设$f$为强凸函数且存在最小值,则$f$的最小值点唯一.

### 凸函数的判定定理

凸函数的一个最基本的判定方式是:先将其限制在任意直线上,然后判断对应的一维函数是否是凸的.

:::info 定理

$f: \mathbb{R}^n \to \mathbb{R}$是凸函数,当且仅当对每个$x \in \mathrm{dom}\,f ,v\in \mathbb{R}^n$, 函数$g: \mathbb{R} \to \mathbb{R}$,

$$
g(t) = f(x+tv), \quad \mathrm{dom}\,g = \{ t | x + tv \in \mathrm{dom}\,f \}
$$

是关于$t$的凸函数.

:::

**例**: $f(X)=-\log \det X$是凸函数,其中$\mathrm{dom}\,f = \mathbb{S}_{++}^n$. 任取$X \succ 0$以及方向$V \in \mathbb{S}^n$,将$f$限制在直线$X+tV$ ($t$满足$X+tV \succ 0$)上, 那么

$$
\begin{aligned}
g(t) = -\log \det (X + tV) 
&= -\log \det X - \log \det (I + tX^{ -\frac{1}{2} }VX^{ -\frac{1}{2} }) \\
&= -\log \det X - \sum_{i=1}^{n}\log(1 + t\lambda_i)
\end{aligned}
$$

其中$\lambda_i$是$X^{ -\frac{1}{2} }VX^{ -\frac{1}{2} }$的第$i$个特征值.

对于每个$X \succ 0$以及方向$V$, $g$关于$t$是凸的,因此$f$是凸的.

<details>

<summary>证明</summary>

**必要性**:设$f(x)$是凸函数,要证$g(t)=f(x+tv)$是凸函数.先说明$\mathrm{dom}\, g$是凸集.

对$\forall t_1, t_2 \in \mathrm{dom}\,g$以及$\theta \in (0,1)$,

$$
x + t_1v \in \mathrm{dom}\,f,\,x+t_2v\in\mathrm{dom}\,f
$$

$\because \mathrm{dom}\,f$是凸集,又有

$$
\begin{aligned}
\theta(x + t_1v) + (1-\theta)(x + t_2v) &= x + \theta t_1v + (1-\theta)t_2v \\
&= x + [\theta t_1 + (1-\theta)t_2]v
\end{aligned}
$$

我们有推论:凸集的线性组合仍是凸集.故可知$x + (\theta t_1 + (1-\theta)t_2)v \in \mathrm{dom}\,f$,

$$
\left. 
    \begin{aligned}
        t_1,t_2 \in \mathrm{dom}\,g \\
        g(t) = f(x+tv)\\
        x + (\theta t_1 + (1-\theta)t_2)v \in \mathrm{dom}\,f
    \end{aligned}
\right\}
\Rightarrow \theta t_1 + (1-\theta)t_2 \in \mathrm{dom}\,g \Rightarrow \mathrm{dom}\,g\text{是凸集}.
$$

此外我们有

$$
\begin{aligned}
g(\theta t_1 + (1-\theta)t_2)
&= f(x + (\theta t_1 + (1-\theta)t_2)v) \\
&= f(\theta(x + t_1v) + (1-\theta)(x + t_2v)) \\
&\leq \theta f(x + t_1v) + (1-\theta)f(x + t_2v) \;\; (\text{凸函数定义}) \\
&= \theta g(t_1) + (1-\theta) g(t_2).
\end{aligned}
$$

结合以上两点可以得到函数$g(t)$是凸函数.

---

**充分性**:先说明$\mathrm{dom}\,f$是凸集.

取$\forall \theta \in [0, 1],\, v = y - x$,以及$t_1=0, t_2=1$.则由$\mathrm{dom}\,g$是凸集可知$\theta t_1 + (1-\theta)t_2 = \theta \cdot 0 + (1-\theta) \cdot 1 \in \mathrm{dom}\,g$

$$
\begin{aligned}
\because x+t_1v &= x + 0 \cdot (y-x) = x \in \mathrm{dom}\,f \\
x+t_2v &= x + 1 \cdot (y-x) = y \in \mathrm{dom}\,f
\end{aligned}
$$
$$
\therefore \theta(x+t_1v) + (1-\theta)(x+t_2v) = \theta x + (1-\theta)y \in \mathrm{dom}\,f\;\text{即}\,\mathrm{dom}\,f\text{是凸集}.
$$

再根据$g(t)=f(x+tv)$的凸性,我们有

$$
\begin{aligned}
g(1-\theta)
&= g(\theta t_1 + (1-\theta)t_2) \\
&\leq \theta g(t_1) + (1-\theta)g(t_2) \\
&= \theta g(0) + (1-\theta)g(1) \\
&= \theta f(x) + (1-\theta)f(y).
\end{aligned}
$$

而又有

$$
g(1-\theta) = f(x + (1-\theta)(y-x)) = f(\theta x + (1-\theta)y), \\
\therefore f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)
$$

这说明$f(x)$是凸函数.故原命题得证.

</details>

### 一阶条件

:::info 定理

对于定义在凸集上的可微函数$f$, $f$是凸函数当且仅当

$$
f(y) \geq f(x) + \nabla f(x)^\top (y-x) \quad \forall x, y \in \mathrm{dom}\,f
$$

:::

<details>

<summary>证明</summary>

**必要性**: 设$f$是凸函数,则对于$\forall x, y \in \mathrm{dom}\,f$以及$t \in (0,1)$,有

$$
\begin{aligned}
&\qquad tf(y) + (1-t)f(x) \geq f(x+t(y-x)) \\
&\Rightarrow tf(y) + f(x) - tf(x) \geq f(x+t(y-x)) \\
&\Rightarrow f(y)  - f(x) \geq \frac{f(x+t(y-x)) - f(x)}{t}
\end{aligned}
$$

令$t \to 0$,由极限保号性可得

$$
f(y)  - f(x) \geq \lim_{t\to 0}\frac{f(x+t(y-x)) - f(x)}{t} = \nabla f(x)^\top (y-x)
$$

这里最后一个等式成立是由于方向导数的性质:

如果函数$f$在点$\mathbf {x}$处可微，则沿着任意非零向量$\mathbf {v}$的方向导数都存在。则有：

$$
\nabla_{\mathbf{v}}{f}(\mathbf{x})=\mathrm{D} f_{\mathbf {x} }(\mathbf{v})=\mathbf{v}\cdot\nabla f(\mathbf{x})
$$

其中$\mathrm{D}f_{\mathbf{x}}$是函数$f$在点$\mathbf{x}$的全微分，为一线性映射；$\nabla$符号表示梯度算子，而“$\cdot$”表示$\mathbb{R}^n$中的内积. (注：在这例子里，如果线性映射$\mathrm{D}f_{\mathbf{x}}$用矩阵表示且选用自然基底的话，$\mathrm {D}f_{\mathbf{x}}=\nabla f({\mathbf{x}})$为$1\times n$的矩阵).

---

**充分性**: 对$\forall x, y \in \mathrm{dom}\,f$以及$\forall t \in (0,1)$, 定义$z = tx+(1-t)y$, 应用两次一阶条件我们有

$$
\begin{aligned}
f(x) \geq f(z) + \nabla f(z)^\top (x-z)\qquad &\text{(1)}\\
f(y) \geq f(z) + \nabla f(z)^\top (y-z)\qquad &\text{(2)}
\end{aligned}
$$

令$(1)$乘上$t$,$(2)$乘上$1-t$,相加得

$$
tf(x) + (1-t)f(y) \geq f(z) + tx + (1-t)y - z = f(z) = f(tx+(1-t)y)
$$

满足凸函数的定义,因此充分性成立.

</details>

### 梯度单调性

:::info 定义

设$f$为可微函数,则$f$为凸函数当且仅当$\mathrm{dom}\,f$为凸集且$\nabla f$为单调映射,

$$
(\nabla f(x) - \nabla f(y))^\top (x - y) \geq 0, \quad \forall x, y \in \mathrm{dom}\,f
$$

:::

<details>

<summary>证明</summary>

**必要性**: 若$f$可微且为凸函数,根据一阶条件,我们有

$$
\begin{aligned}
f(x) \geq f(x) + \nabla f(x)^\top (y-x)\qquad &\text{(1)}\\
f(y) \geq f(y) + \nabla f(y)^\top (x-y)\qquad &\text{(2)}
\end{aligned}
$$

$(1) + (2)$得

$$
\begin{aligned}
f(x) + f(y) &\geq f(x) + f(y) + \nabla f(x)^\top (y-x) + \nabla f(y)^\top (x-y) \\
&\Rightarrow 0 \geq \nabla f(x)^\top (y-x) - \nabla f(y)^\top (y-x) \\
&\Rightarrow 0 \geq (\nabla f(x)^\top - \nabla f(y)^\top)(y-x) \\
&\Rightarrow (\nabla f(x) - \nabla f(y))^\top(x-y) \geq  0
\end{aligned}
$$

---

**充分性**: 若$\nabla f$为单调映射,构造一元辅助函数

$$
g(t) = f(x+t(y-x)), \quad g^\prime(t) = \nabla f(x+t(y-x))^\top(y-x)
$$

由$\nabla f$的单调性可知$g^\prime(t) \geq g^\prime(0), \; \forall t \geq 0$.因此

$$
f(y) = g(1) = g(0) + \int_0^1g^\prime(t) dt \geq g(0) + g^\prime(0) = f(x) + \nabla f(x)^\top(y-x).
$$

</details>

### 上方图

:::info 定理

函数$f(x)$为凸函数当且仅当其上方图$epi\; f$是凸集.

:::

<details>

<summary>证明</summary>

**必要性**: 若$f(x)$为凸函数,则对$\forall (x_1, y_1), (x_2, y_2) \in epi\; f, t \in [0,1]$,

$$
ty_1 + (1-t)y_2 \geq tf(x_1) + (1-t) f(x_2) \geq f(tx_1 + (1-t)x_2) \\
\therefore (tx_1+(1-t)x_2,ty_1 + (1-t)y_2) \in epi\;f\\
\Rightarrow f(tx_1+(1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)
$$

---

**充分性**: 若$epi\;f$是凸集,则对$\forall x_1, x_2 \in \mathrm{dom}\,f , t \in [0,1]$,

$$
\begin{aligned}
(tx_1+(1-t)x_2, tf(x_1)+(1-t)f(x_2)) \in epi\;f \\
\Rightarrow f(tx_1+(1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)
\end{aligned}
$$

</details>

### 二阶条件

:::info 定理

设$f$为定义在凸集上的二阶连续可微函数

$f$是凸函数当且仅当

$$
\nabla^2f(x) \succeq 0, \forall x \in \mathrm{dom}\;f
$$

如果$\nabla^2f(x) \succ 0 , \forall x \in \mathrm{dom}\;f$,则$f$是严格凸函数.

:::

**例**:

二次函数$f(x) = \frac{1}{2}x^\top Px + q^\top x + r$,其中$(P \in \mathbb{S}^n)$

$$
\nabla f(x) = Px + q, \qquad \nabla^2f(x) = P
$$

$f$是凸函数当且仅当$P \succeq 0$

<details>

<summary>证明</summary>

**必要性**: 反设$f(x)$在点$x$处的海瑟矩阵$\nabla^2f(x)\not \succeq 0$, 即存在非零向量$v \in \mathbb{R}^n$使得$v^\top\nabla^2f(x)v < 0$. 根据佩亚诺(Peano)余项的泰勒展开,

$$
\begin{aligned}
f(x + tv) = f(x) + t\nabla f(x)^\top v + \frac{t^2}{2}v^\top\nabla^2f(x)v + o(t^2) \\
\frac{f(x + tv) - f(x) - t\nabla f(x)^\top v}{t^2} = \frac{1}{2}v^\top\nabla^2f(x)v + o(1)
\end{aligned}
$$

当$t$充分小时,

$$
\frac{f(x + tv) - f(x) - t\nabla f(x)^\top v}{t^2} < 0,
$$

这显然和一阶条件矛盾,因此必有$\nabla^2f(x) \succeq 0$成立.

---

**充分性**: 设$f(x)$满足二阶条件$\nabla^2f(x) \succeq 0$,对$\forall x, y \in \mathrm{dom}\;f$,根据泰勒展开,

$$
f(y) = f(x) + \nabla f(x)^\top(y-x) + \frac{1}{2}(y-x)^\top \nabla^2f(x+t(y-x))(y-x),
$$

其中$t \in (0, 1)$是和$x,y$有关的常数. 由半正定性可知对$\forall x,y \in \mathrm{dom}\;f$有

$$
f(y) \geq f(x) + \nabla f(x)^\top(y-x).
$$

由凸函数判定的一阶条件知$f$为凸函数. 进一步, 若$\nabla^2f(x) > 0$, 上式中不等号严格成立($x \neq y$). 利用一阶条件的充分性的证明过程可得$f(x)$为严格凸函数.

对$\forall x, y \in \mathrm{dom}\;f,\forall t\in(0,1)$,设$z = tx + (1-t)y \in \mathrm{dom}\;f$,则

$$
f(x) = f(z) + \nabla f(z)^\top(x-z) + \frac{1}{2}(x-z)^\top \nabla^2f(z+t(x-z))(x-z) \\
f(y) = f(z) + \nabla f(z)^\top(y-z) + \frac{1}{2}(y-z)^\top \nabla^2f(z+t(y-z))(y-z)
$$
$$
\because \nabla^2f(x) > 0 \\
\Rightarrow 
\begin{cases}
f(x) > f(z) + \nabla f(z)^\top(x-z) & \text{(1)} \\
f(y) > f(z) + \nabla f(z)^\top(y-z) & \text{(2)}
\end{cases}
$$

$t\times(1) + (1-t)\times(2)$得

$$
\begin{aligned}
tf(x) + (1-t)f(y) &> f(z) + \nabla f(z)^\top [t(x-z) + (1-t)(y-z)] \\
&= f(z) + \nabla f(z)^\top [tx + (1-t)y - z] \\
&= f(tx + (1-t)y) \\
\end{aligned}
$$
$$
\therefore\text{函数} f \text{是严格凸函数}.
$$

</details>

### 二阶条件的应用

:::danger 注意

该章节的前置知识包括矩阵论的矩阵求导部分.

:::

**最小二乘函数**:

$$
\begin{aligned}
f(x) &= ||Ax-b||_2^2 \\
\nabla f(x) &= 2A^\top (Ax-b) \\
\nabla^2 f(x) &= 2A^\top A
\end{aligned}
$$

**Quadratic-over-linear函数**:

$$
\begin{aligned}
f(x,y) &= \frac{x^2}{y} \\
\nabla^2 f(x, y) &= \frac{2}{y^3}
\begin{bmatrix}
y \\
-x
\end{bmatrix}\begin{bmatrix}
y \\
-x
\end{bmatrix}^\top \succeq 0 \\
\text{是区域}\{ &(x,y) | y > 0 \}\text{上的凸函数}.
\end{aligned}
$$

**Log-sum-exp函数**:

$$
\begin{aligned}
f(x) &= \log \sum_{k=1}^{n}\exp x_k \; \text{是凸函数}.
\nabla^2f(x) &= \frac{1}{\mathbf{1}^\top z}\mathrm{diag}(z) - \frac{1}{(\mathbf{1}^\top z)^2}zz^\top \quad (z_k = \exp x_k)
\end{aligned}
$$

要证明$\nabla^2f(x)\succeq 0$,我们只需证明对$\forall v$使得$v^\top\nabla^2f(x)v \geq 0$,即

$$
v^\top\nabla^2f(x)v = \frac{ (\sum_k z_kv_k^2)(\sum_k z_k) - (\sum_k v_kz_k)^2 }{ (\sum_k z_k)^2 } \geq 0
$$

由柯西不等式,得

$$
(\sum_k v_kz_k)^2 \leq (\sum_k z_kv_k^2)(\sum_k z_k)
$$

因此$f$是凸函数.

**几何平均**:

$$
f(x) = (\prod_{k=1}^{n}x_k)^\frac{1}{n} \quad (x \in \mathbb{R}_{++}^n) \text{是凹函数}.
$$

### Jensen不等式/琴生不等式

**基础Jensen不等式**:

设$f$是凸函数,则对于$0 \leq \theta \leq 1$,有

$$
f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)
$$

**概率Jensen不等式**:

设$f$是凸函数,则对于任意随机变量$z$,有

$$
f(\mathbb{E}[z]) \leq \mathbb{E}_z[f(z)]
$$

基础Jensen不等式可以视为概率Jensen不等式在两点分布下的特殊情况

$$
\mathrm{prob} (z=x) = \theta, \quad \mathrm{prob}(z=y) = 1- \theta
$$

## 保凸的运算

:::info 验证一个函数是否为凸函数的方法

1. 用定义验证(通常将函数限制在一条直线上)
2. 利用一阶条件和二阶条件
3. 直接研究$f$的上方图$epi\;f$
4. 说明$f$可由简单的凸函数通过一些保凸的运算得到
   - 非负加权和
   - 与仿射函数的复合
   - 逐点取最大值
   - 与标量、向量函数的复合
   - 取下确界
   - 透视函数

:::

### 非负加权和与仿射函数的复合

:::info

**非负数乘**: 若$f$是凸函数,则$\alpha f$是凸函数,其中$\alpha\geq0$.

**求和**: 若$f1,f2$是凸函数,则$f1 + f2$是凸函数.

**与仿射函数的复合**: 若$f$是凸函数,则$f(Ax + b)$是凸函数.

:::

**例**: 

线性不等式的对数障碍函数:

$$
f(x) = -\sum_{i=1}^{m}\log (b_i - a_i^\top x), \mathrm{dom}\;f = \{ x | a_i^\top x < b_i, \, i=1, \dots, m \}
$$

仿射函数的(任意)范数: $f(x)=||Ax+b||$

### 逐点取最大值

:::info

若$f_1, \dots , f_m$是凸函数,则$f(x)=\max\{ f_1(x), \dots, f_m(x) \}$是凸函数.

:::

**例**:

分段线性函数: 

$$
f(x)=\max_{i=1,\dots,m}(a_i^\top x + b_i) \; \text{是凸函数.}
$$

$x \in \mathbb{R}^n$的前$r$个最大分量之和:

$$
f(x) = x_{[1]} + x_{[2]} + \cdots + x_{[r]} \; \text{是凸函数.}
$$

其中$x_{[i]}$为$x$的从大到小排列的第$i$个分量,

事实上,$f(x)$可以写成如下多个线性函数取最大值的形式:

$$
f(x) = \max \{ x_{i_1} + x_{i_2} + \cdots + x_{i_r} | 1 \leq i_1 < i_2 < \cdots < i_r \leq n \}
$$

### 逐点取上界

:::info

若对每个$y \in \mathcal{A}$,$f(x,y)$是关于$x$的凸函数,则

$$
g(x) = \sup_{y\in \mathcal{A}}f(x,y) \; \text{是凸函数.}
$$

:::

**例**:

集合$C$的支撑函数: 

$$
S_C(x) = \sup_{y\in C}y^\top x \; \text{是凸函数.}
$$

集合$C$点到给定点$x$的最远距离:

$$
f(x) = \sup_{y\in C}||x-y||
$$

对称矩阵$X\in \mathbb{S}^n$的最大特征值:

$$
\lambda_{\max}(X) = \sup_{||y||_2=1}y^\top Xy
$$

### 与标量函数的复合

:::info

给定函数$g: \mathbb{R}^n \to \mathbb{R}$和$h: \mathbb{R} \to \mathbb{R}$,

$$
f(x) = h(g(x))
$$

若$g$是凸函数,$h$是凸函数,$\hat{h}$单调不减,那么$f$是凸函数.

若$g$是凹函数,$h$是凸函数,$\hat{h}$单调不增,那么$f$是凸函数.

:::

对$n=1$,$g,h$均可微的情形,我们给出简证

$$
f^{\prime\prime}(x) = h^{\prime\prime}(g(x))g^\prime(x)^2 + h^\prime(g(x))g^{\prime\prime}(x)
$$

注意: 必须是$\hat{h}$满足单调不减/不增的条件;如果仅是$h$满足单调不减/不增的条件,存在反例.

**推论**:

 - 如果$g$是凸函数,则$\exp g(x)$是凸函数.
 - 如果$g$是正值凹函数,则$\frac{1}{g(x)}$是凸函数.

### 与向量函数的复合

:::info

给定函数$g: \mathbb{R}^n \to \mathbb{R}$和$h: \mathbb{R} \to \mathbb{R}$,

$$
f(x) = h(g(x)) = h(g_1(x), g_2(x), \dots, g_k(x))
$$

若$g$是凸函数,$h$是凸函数,$\hat{h}$关于每个分量单调不减,那么$f$是凸函数.

若$g$是凹函数,$h$是凸函数,$\hat{h}$关于每个分量单调不增,那么$f$是凸函数.

:::

对$n=1$,$g,h$均可微的情形,我们给出简证

$$
f^{\prime\prime}(x) = g^\prime(x)^\top\nabla^2h(g(x))g^\prime(x) + \nabla h(g(x))^\top g^{\prime\prime}(x)
$$

**推论**:

 - 如果$g_i$是正值凹函数,则$\sum_{i=1}^{m}\log g_i(x)$是凹函数.
 - 如果$g_i$是凸函数,则$\log\sum_{i=1}^{m}\exp g_i(x)$是凸函数.

### 取下确界

:::info

若$f(x,y)$关于$(x,y)$征途是凸函数,$C$是凸集,则

$$
g(x) = \inf_{y\in C}f(x,y) \; \text{是凸函数.}
$$

:::

**例**:

1. 考虑函数$f(x,y) = x^\top Ax + 2x^\top By + y^\top Cy$,海瑟矩阵满足
   $$
   \begin{bmatrix}
   A & B \\
   B^\top & C
   \end{bmatrix}
   \succeq 0, \quad C \succ 0,
   $$
   则$f(x,y)$为凸函数.对$y$求最小值得
   $$
   g(x) = \inf_y f(x,y) = x^\top(A-BC^{-1}B^\top)x,
   $$
   因此$g$是凸函数.进一步地,$A$的Schur补$A-BC^{-1}B^\top \succeq 0$


2. 点$x$到凸集$S$的距离$\mathrm{dist}(x, S) = \inf_{y\in S}||x-y||$是凸函数.

### 透视函数

:::info

定义$f: \mathbb{R}^n \to \mathbb{R}$的透视函数$g: \mathbb{R}^n \times \mathbb{R} \to \mathbb{R}$,

$$
g(x,t) = tf(\frac{x}{t}), \quad \mathrm{dom}\;g = \Big\{ (x,t) | \frac{x}{t} \in \mathrm{dom}\;f , t > 0 \Big\}
$$

若$f$是凸函数,则$g$是凸函数.

:::

**例**:

1. $f(x) = x^\top x$是凸函数,因此$g(x,t) = \frac{x^\top x}{t}$是区域$\{ (x,t) | t > 0 \}$上的凸函数.
2. $f(x) = -\log x$是凸函数,因此相对熵函数$g(x, t) = t\log t - t\log x$是$\mathbb{R}^2_{++}$上的凸函数
3. 若$f$是凸函数,那么
   $$
   g(x) = (c^\top x + d)f(\frac{Ax+b}{c^\top x + d})
   $$
   是区域$\{ x | c^\top x + d > 0, \frac{Ax + b}{c^\top x + d} \in \mathrm{dom}\;f \}$上的凸函数.

### 共轭函数

:::info

适当函数$f$的共轭函数定义为

$$
f^*(y) = \sup_{x \in \mathrm{dom}\;f}(y^\top x - f(x))
$$

:::

$f^*$恒为凸函数,无论$f$是否是凸函数.

**例**:

1. 负对数$f(x) = -\log x$
   $$
   f^*(y) = \sup_{x > 0}(xy + \log x) = 
   \begin{cases}
   -1 - \log (-y) & y < 0 \\
   \infty & \mathrm{Otherwise}.
   \end{cases}
   $$
2. 强凸二次函数$f(x) = \frac{1}{2}x^\top Qx, \; Q\in\mathbb{S}^n_{++}$
   $$
   \begin{aligned}
   f^*(y) 
   &= \sup_x(y^\top x - \frac{1}{2}x^\top Qx) \\
   &= \frac{1}{2}y^\top Q^{-1}y
   \end{aligned}
   $$

## 凸函数的推广

### 拟凸函数

:::info

设$f:\mathbb{R}^n\to \mathbb{R}$,如果$\mathrm{dom}\;f$是凸集,并且下水平集

$$
S_\alpha = \{ x \in \mathrm{dom}\;f | f(x) \leq \alpha \}
$$

对任意$\alpha$都是凸的

:::

 - 若$f$是拟凸的,则称$-f$是拟凹的.
 - 若$f$既是拟凸的,又是拟凹的,则称$f$是拟线性的.

### 拟凸、凹函数的例子

:::info

1. $\sqrt{|x|}$是$\mathbb{R}$上的拟凸函数.
2. $\mathrm{ceil}(x) = \inf \{ z\in \mathbb{Z} | z \geq x \}$是拟线性的.
3. $\log x$是$\mathbb{R}_{++}$上的拟线性函数.
4. $f(x_1, x_2) = x_1x_2$是$\mathbb{R}_{++}^2$上的拟凹函数.
5. 分式线性函数
   $$
   f(x)=\frac{a^\top x+b}{c^\top x+d}, \; \mathrm{dom}\;f = \{ x | c^\top x + d > 0 \} \; \text{是拟线性的}.
   $$
6. 距离比值函数
   $$
   f(x)=\frac{||x-a||_2}{||x-b||_2}, \; \mathrm{dom}\;f = \{ x | \;  ||x-a||_2 \leq ||x-b||_2 \} \; \text{是拟凸的}.
   $$

:::

**例**: 内部回报率(IRR)

现金流$x = (x_0,\dots,x_n);\;x_i$是$i$时段支付的现金.

假定$x_0 < 0, \, x_0 + x_1 + \cdots + x_n > 0$.

现值(Present Value)的表达式为

$$
\mathrm{PV}(x,r) = \sum_{i=0}^{n}(1+r)^{-i}x_i
$$

其中$r$为利率.

内部回报率(IRR)是最小的使得$\mathrm{PV}(x,r) = 0$的利率$r$

$$
\mathrm{IRR}(x) = \inf \{ r \geq 0 | \mathrm{PV}(x,r) = 0 \}
$$

IRR是拟凹的,因为它的上水平集是开半空间的交

$$
\mathrm{IRR}(x) \geq R \Leftrightarrow \sum_{i=0}^{n}(1+r)^{-i} x_i > 0 \quad \mathrm{for} \;0\leq r < R
$$

### 拟凸函数的性质

**类Jensen不等式**:

对拟凸函数$f$,
$$
0\leq\theta\leq1 \Rightarrow f(\theta x + (1-\theta)y) \leq \max \{ f(x), f(y) \}
$$

**一阶条件**:

定义在凸集上的可微函数$f$是拟凸的,当且仅当

$$
f(y)\leq f(x) \Rightarrow \nabla f(x)^\top(y-x) \leq 0
$$

注: 拟凸函数的和不一定是拟凸函数.

### 对数凸函数

:::info

如果正值函数$f$满足$\log f$是凸函数,则$f$称为对数凸函数,即

$$
f(\theta x + (1-\theta)y) \leq f(x)^\theta f(y)^{1-\theta} \quad \mathrm{for} \; 0\leq\theta\leq1.
$$

如果$\log f$是凹函数,则$f$称为对数凹函数.

:::

幂函数: 当$a\leq 0$时,$x^{a}$是$\mathbb{R}_{++}$上的对数凸函数; 当$a\geq 0$,$x^a$是$\mathbb{R}_{++}$上的对数凹函数.

许多常见的概率密度函数是对数凹函数,例如正态分布

$$
f(x) = \frac{1}{ \sqrt{(2\pi)^n\det\sum} }e^{ -\frac{1}{2}(x-\overline{x})^\top\sum^{-1}(x-\overline{x}) }
$$

高斯分布的累计分布函数$\Phi$是对数凹函数.

$$
\Phi(x) = \frac{1}{ \sqrt{2\pi} }\int_{-\infty}^{x}e^{-\frac{1}{2}u^2}du
$$

### 对数凸、凹函数的性质

:::info

定义在凸集上的二阶可微函数$f$是对数凹的,当且仅当

$$
f(x)\nabla^2f(x)\preceq\nabla f(x)\nabla f(x)^\top
$$

对任意$x\in\mathrm{dom}\;f$成立.

对数凹函数的乘积仍为对数凹函数.

对数凹函数的和不一定为对数凹函数.

若$f:\mathbb{R}^n \times \mathbb{R}^m\to \mathbb{R}$是对数凹函数,那么

$$
g(x)=\int f(x,y)dy
$$

是对数凹函数.

**对数凹函数的积分**: 

对数凹函数$f,g$的卷积$f*g$是对数凹函数

$$
(f*g)(x)=\int f(x-y)g(y)dy
$$

若$C \subseteq \mathbb{R}^n$是凸集,并且随机变量$y$的概率密度函数是对数凹函数,则

$$
f(x) = \mathrm{prob}(x+y \in C)\;\text{是对数凹函数}.
$$

证明: $f(x)$可表示为两个对数凹函数乘积的积分

$$
f(x) = \int g(x+y)p(y)dy,\quad g(u) = 
\begin{cases}
1 & u\in C \\
0 & u\not\in C
\end{cases}
$$

其中$p$是$y$的概率密度函数.

:::

**例**: 生成函数

$$
Y(x) = \mathrm{prob}(x+ w\in S)
$$

 - $x\in\mathbb{R}^n$: 产品的标称参数(Nominal Parameter).
 - $w\in\mathbb{R}^n$: 制成品的参数是随机变量.
 - $S$: 接受集.

若$S$是凸集,并且随机变量$w$的概率密度函数是对数凹函数,则

 - $Y$是拟凹函数
 - 生成区域$\{ x | Y(x) \geq \alpha \}$是凸集.

### 广义不等式意义下的凸函数

:::info

$f:\mathbb{R}^n\to \mathbb{R}^m$称为$K$-凸函数: 如果$\mathrm{dom}\;f$是凸集,并且

$$
f(\theta x+(1-\theta)y) \preceq_K \theta f(x)+ (1-\theta)f(y)
$$

对$\forall x,y \in \mathrm{dom}\;f,\,0\leq\theta\leq1$成立

:::

**例**:

$f:\mathbb{S}^m\to\mathbb{S}^m, f(X)=X^2$是$\mathbb{S}^m_{+}$-凸函数

证明:

对固定的$z\in \mathbb{R}^m,\, z^\top X^2z=||Xz||_2^2$关于$X$是凸函数,即

$$
z^\top (\theta X + (1-\theta)Y)^2z\leq \theta z^\top X^2z + (1-\theta)z^\top Y^2z
$$

对$\forall X,Y \in \mathbb{S}^m,\, 0\leq\theta\leq1$成立.

$$
\therefore (\theta X + (1-\theta)Y)^2 \preceq \theta X^2 + (1-\theta)Y^2
$$