# softmax回归的代码实现

这一节我们来动手实现softmax回归。首先导入本节实现所需的包或模块。

``` python
import tensorflow as tf
import numpy as np
print(tf.__version__)
```

输出：

```
2.0.0
```

## 手动实现

### 获取和读取数据

我们将使用Fashion-MNIST数据集，并设置批量大小为256。

``` python
from tensorflow.keras.datasets import fashion_mnist

batch_size=256
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = tf.cast(x_train, tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
x_test = tf.cast(x_test,tf.float32) / 255 #在进行矩阵相乘时需要float型，故强制类型转换为float型
train_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
```

### 初始化模型参数

跟线性回归中的例子一样，我们将使用向量表示每个样本。已知每个样本输入是高和宽均为28像素的图像。模型的输入向量的长度是 $28 \times 28 = 784$：该向量的每个元素对应图像中每个像素。由于图像有10个类别，单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为$784 \times 10$和$1 \times 10$的矩阵。`Variable`来标注需要记录梯度的向量。

``` python
num_inputs = 784
num_outputs = 10
W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01, dtype=tf.float32))
b = tf.Variable(tf.zeros(num_outputs, dtype=tf.float32))
```

### 实现 softmax 运算

在介绍如何定义 softmax 回归之前，我们先描述一下对如何对多维`Tensor`按维度操作。在下面的例子中，给定一个`Tensor`矩阵`X`。我们可以只对其中同一列（`axis=0`）或同一行（`axis=1`）的元素求和，并在结果中保留行和列这两个维度（`keepdims=True`）。

``` python
X = tf.constant([[1, 2, 3], [4, 5, 6]])
tf.reduce_sum(X, axis=0, keepdims=True), tf.reduce_sum(X, axis=1, keepdims=True)
```

输出：

```
(<tf.Tensor: id=462401, shape=(1, 3), dtype=int32, numpy=array([[5, 7, 9]], dtype=int32)>,
 <tf.Tensor: id=462403, shape=(2, 1), dtype=int32, numpy=
 array([[ 6],
        [15]], dtype=int32)>)
```

下面我们就可以定义前面小节里介绍的softmax运算了。在下面的函数中，矩阵`logits`的行数是样本数，列数是输出个数。为了表达样本预测各个输出的概率，softmax运算会先通过`exp`函数对每个元素做指数运算，再对`exp`矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。这样一来，最终得到的矩阵每行元素和为1且非负。因此，该矩阵每行都是合法的概率分布。softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。

``` python
def softmax(logits, axis=-1):
    return tf.exp(logits)/tf.reduce_sum(tf.exp(logits), axis, keepdims=True)
```

可以看到，对于随机输入，我们将每个元素变成了非负数，且每一行和为1。

``` python
X = tf.random.normal(shape=(2, 5))
X_prob = softmax(X)
X_prob, tf.reduce_sum(X_prob, axis=1)
```

输出：

```
(<tf.Tensor: id=462414, shape=(2, 5), dtype=float32, numpy=
 array([[0.07188913, 0.19016613, 0.21624805, 0.40005335, 0.12164329],
        [0.20424965, 0.22559293, 0.13348413, 0.2243966 , 0.21227665]],
       dtype=float32)>,
 <tf.Tensor: id=462416, shape=(2,), dtype=float32, numpy=array([1.        , 0.99999994], dtype=float32)>)
```

###  定义模型

有了softmax运算，我们可以定义上节描述的softmax回归模型了。这里通过`reshpe`函数将每张原始图像改成长度为`num_inputs`的向量。

``` python
def net(X):
    logits = tf.matmul(tf.reshape(X, shape=(-1, W.shape[0])), W) + b
    return softmax(logits)
```

### 定义损失函数

上一节中，我们介绍了softmax回归使用的交叉熵损失函数。为了得到标签的预测概率，我们可以使用`boolean_mask`函数和`one_hot`函数。在下面的例子中，变量`y_hat`是2个样本在3个类别的预测概率，变量`y`是这2个样本的标签类别。通过使用`gather`函数，我们得到了2个样本的标签的预测概率。与3.4节（softmax回归）数学表述中标签类别离散值从1开始逐一递增不同，在代码中，标签类别的离散值是从0开始逐一递增的。

``` python
y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = np.array([0, 2], dtype='int32')
tf.boolean_mask(y_hat, tf.one_hot(y, depth=3))
```

输出：

```
<tf.Tensor: id=462449, shape=(2,), dtype=float64, numpy=array([0.1, 0.5])>
```

下面实现了3.4节（softmax回归）中介绍的交叉熵损失函数。（注：由于在 Tensorflow 涉及运算类型转换的问题，使用`cast`函数对张量进行类型转换。）

``` python
def cross_entropy(y_hat, y):
    y = tf.cast(tf.reshape(y, shape=[-1, 1]),dtype=tf.int32)
    y = tf.one_hot(y, depth=y_hat.shape[-1])
    y = tf.cast(tf.reshape(y, shape=[-1, y_hat.shape[-1]]),dtype=tf.int32)
    return -tf.math.log(tf.boolean_mask(y_hat, y)+1e-8)
```

### 计算分类准确率

给定一个类别的预测概率分布`y_hat`，我们把预测概率最大的类别作为输出类别。如果它与真实类别`y`一致，说明这次预测是正确的。分类准确率即正确预测数量与总预测数量之比。

为了演示准确率的计算，下面定义准确率`accuracy`函数。其中`tf.argmax(y_hat, axis=1)`返回矩阵`y_hat`每行中最大元素的索引，且返回结果与变量`y`形状相同。相等条件判断式`(tf.argmax(y_hat, axis=1) == y)`是一个数据类型为`bool`的`Tensor`，实际取值为：0（相等为假）或 1（相等为真）。

``` python
def accuracy(y_hat, y):
    return np.mean((tf.argmax(y_hat, axis=1) == y))
```

让我们继续使用在演示`boolean_mask`函数时定义的变量`y_hat`和`y`，并将它们分别作为预测概率分布和标签。可以看到，第一个样本预测类别为2（该行最大元素0.6在本行的索引为2），与真实标签0不一致；第二个样本预测类别为2（该行最大元素0.5在本行的索引为2），与真实标签2一致。因此，这两个样本上的分类准确率为0.5。

``` python
accuracy(y_hat, y)
```

输出：

```
0.5
```

类似地，我们可以评价模型`net`在数据集`data_iter`上的准确率。

``` python
# 描述,对于tensorflow2中，比较的双方必须类型都是int型，所以要将输出和标签都转为int型
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for _, (X, y) in enumerate(data_iter):
        y = tf.cast(y,dtype=tf.int64)
        acc_sum += np.sum(tf.cast(tf.argmax(net(X), axis=1), dtype=tf.int64) == y)
        n += y.shape[0]
    return acc_sum / n
```

因为我们随机初始化了模型`net`，所以这个随机模型的准确率应该接近于类别个数 10 的倒数即 0.1。

``` python
print(evaluate_accuracy(test_iter, net))
```

输出：

```
0.0834
```

###  训练模型

训练softmax回归的实现跟 3.2（线性回归的从零开始实现）一节介绍的线性回归中的实现非常相似。我们同样使用小批量随机梯度下降来优化模型的损失函数。在训练模型时，迭代周期数`num_epochs`和学习率`lr`都是可以调的超参数。改变它们的值可能会得到分类更准确的模型。

``` python
num_epochs, lr = 5, 0.1
# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with tf.GradientTape() as tape:
                y_hat = net(X)
                l = tf.reduce_sum(loss(y_hat, y))
            grads = tape.gradient(l, params)
            if trainer is None:
                # 如果没有传入优化器，则使用原先编写的小批量随机梯度下降
                for i, param in enumerate(params):
                    param.assign_sub(lr * grads[i] / batch_size)
            else:
                # tf.keras.optimizers.SGD 直接使用是随机梯度下降 theta(t+1) = theta(t) - learning_rate * gradient
                # 这里使用批量梯度下降，需要对梯度除以 batch_size, 对应原书代码的 trainer.step(batch_size)
                trainer.apply_gradients(zip([grad / batch_size for grad in grads], params))  
                
            y = tf.cast(y, dtype=tf.float32)
            train_l_sum += l.numpy()
            train_acc_sum += tf.reduce_sum(tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, dtype=tf.int64), dtype=tf.int64)).numpy()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'% (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

trainer = tf.keras.optimizers.SGD(lr)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
```

输出：

```
epoch 1, loss 0.8969, train acc 0.736, test acc 0.813
epoch 2, loss 0.5987, train acc 0.806, test acc 0.826
epoch 3, loss 0.5524, train acc 0.820, test acc 0.832
epoch 4, loss 0.5297, train acc 0.826, test acc 0.834
epoch 5, loss 0.5139, train acc 0.830, test acc 0.836
```

###  预测

训练完成后，现在就可以演示如何对图像进行分类了。给定一系列图像（第三行图像输出），我们比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）。

``` python
import matplotlib.pyplot as plt
X, y = iter(test_iter).next()

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    # 这⾥的_表示我们忽略（不使⽤）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12)) # 这里注意subplot 和subplots 的区别
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(tf.reshape(img, shape=(28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

true_labels = get_fashion_mnist_labels(y.numpy())
pred_labels = get_fashion_mnist_labels(tf.argmax(net(X), axis=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:9], titles[0:9])
```

![img](src/softmax-regression-code/3.6_output1.png)

## 使用框架实现

### 获取和读取数据

我们仍然使用Fashion-MNIST数据集和上一节中设置的批量大小。

``` python
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

对数据进行处理，归一化，便于训练

```python
x_train = x_train / 255.0
x_test = x_test / 255.0
```

### 定义和初始化模型

在3.4节（softmax回归）中提到，softmax回归的输出层是一个全连接层。因此，我们添加一个输出个数为10的全连接层。 第一层是Flatten，将28 * 28的像素值，压缩成一行 (784, ) 第二层还是Dense，因为是多分类问题，激活函数使用softmax

``` python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

### softmax和交叉熵损失函数

如果做了上一节的练习，那么你可能意识到了分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定。因此，Tensorflow2.0的keras API提供了一个loss参数。它的数值稳定性更好。

``` python
loss = 'sparse_categorical_crossentropy'
```

### 定义优化算法

我们使用学习率为0.1的小批量随机梯度下降作为优化算法。

``` python
optimizer = tf.keras.optimizers.SGD(0.1)
```

###  训练模型

接下来，我们使用上一节中定义的训练函数来训练模型。

``` python
model.compile(optimizer=tf.keras.optimizers.SGD(0.1),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=5,batch_size=256)
```

输出：

```
Train on 60000 samples
Epoch 1/5
60000/60000 [==============================] - 1s 20us/sample - loss: 0.7941 - accuracy: 0.7408
Epoch 2/5
60000/60000 [==============================] - 1s 11us/sample - loss: 0.5729 - accuracy: 0.8112
Epoch 3/5
60000/60000 [==============================] - 1s 11us/sample - loss: 0.5281 - accuracy: 0.8241
Epoch 4/5
60000/60000 [==============================] - 1s 11us/sample - loss: 0.5038 - accuracy: 0.8296
Epoch 5/5
60000/60000 [==============================] - 1s 11us/sample - loss: 0.4866 - accuracy: 0.8351
```

接下来，比较模型在测试数据集上的表现情况

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Acc:',test_acc)
```

输出：

```
 - 1s 55us/sample - loss: 0.4347 - accuracy: 0.8186
Test Acc: 0.8186
```

## 小结

* 可以使用softmax回归做多类别分类。与训练线性回归相比，你会发现训练softmax回归的步骤和它非常相似：获取并读取数据、定义模型和损失函数并使用优化算法训练模型。事实上，绝大多数深度学习模型的训练都有着类似的步骤。
