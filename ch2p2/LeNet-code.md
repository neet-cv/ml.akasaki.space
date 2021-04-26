# LeNet代码实现

这里不会包含任何讲解的内容，讲解的内容请参考[第一个卷积神经网络(LeNet)](./LeNet.md)

## 比较标准的写法（85行）

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, initializers

(training_x, training_y), (testing_x, testing_y) = datasets.fashion_mnist.load_data()
training_x = (training_x.astype('float32') / 255.)
testing_x = (testing_x.astype('float32') / 255.)
training_x = tf.reshape(training_x, (training_x.shape[0], training_x.shape[1], training_x.shape[2], 1))
testing_x = tf.reshape(testing_x, (testing_x.shape[0], testing_x.shape[1], testing_x.shape[2], 1))
batch_size = 100

training_dataset = tf.data.Dataset.from_tensor_slices((training_x, training_y))
training_dataset = training_dataset.batch(batch_size)
testing_dataset = tf.data.Dataset.from_tensor_slices((testing_x, testing_y))
testing_dataset = testing_dataset.batch(batch_size)

class LeNetModel(tf.keras.Model):
    def __init__(self):
        super(LeNetModel, self).__init__()
        # 1.第一个卷积层
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(5, 5), padding='SAME', activation=tf.nn.sigmoid, use_bias=True, bias_initializer=initializers.Zeros)
        # 2.最大池化
        self.maxpool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
        # 3.第二个卷积层
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='SAME', activation=tf.nn.sigmoid, use_bias=True, bias_initializer=initializers.Zeros)
        # 4.最大池化
        self.maxpool2 = self.maxpool1 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
        # 5.打平以便进入全连接
        self.flatten = layers.Flatten()
        # 6.第一个全连接层
        self.fc1 = layers.Dense(units=512, activation=tf.nn.sigmoid, use_bias=True, bias_initializer=initializers.Zeros)
        # 7.第一个全连接层
        self.fc2 = layers.Dense(units=10, activation=tf.nn.sigmoid, use_bias=True, bias_initializer=initializers.Zeros)

    # 定义前向传播的方法
    def call(self, inputs, training=None, mask=None):
        result = self.conv1(inputs)
        result = self.maxpool1(result)
        result = self.conv2(result)
        result = self.maxpool2(result)
        result = self.flatten(result)
        result = self.fc1(result)
        result = self.fc2(result)
        return result
# 创建一个模型实例
model = LeNetModel()

loss_fun = tf.losses.SparseCategoricalCrossentropy(name='loss_fun')
train_loss = tf.metrics.Mean(name='train_loss')
train_acc = tf.metrics.SparseCategoricalAccuracy(name='train_acc')
test_loss = tf.metrics.Mean(name='train_loss')
test_acc = tf.metrics.SparseCategoricalAccuracy(name='train_acc')
optimizer = tf.optimizers.Adam()


@tf.function
def training_step(images, labels):
    with tf.GradientTape() as tape:
        pred = model(images)
        loss = loss_fun(labels, pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_acc(labels, pred)


@tf.function
def verify_on_test(images, labels):
    pred = model(images)
    loss = loss_fun(labels, pred)
    test_loss(loss)
    test_acc(labels, pred)


epochs = 40
for ep in range(epochs):
    for images, labels in training_dataset:
        training_step(images, labels)
    for images, labels in testing_dataset:
        verify_on_test(images, labels)
    template = 'Epoch{}, loss:{}, Acc:{}%, test_loss:{}, test_acc:{}%'
    print(template.format(ep + 1,
                          train_loss.result(),
                          train_acc.result() * 100.,
                          test_loss.result(),
                          test_acc.result() * 100.))
```

## 使用tf.keras高层API的快捷写法（29行）

```python
import tensorflow as tf
from tensorflow.keras import datasets

(training_x, training_y), (testing_x, testing_y) = datasets.fashion_mnist.load_data()
training_x = (training_x.astype('float32') / 255.)
testing_x = (testing_x.astype('float32') / 255.)
training_x = tf.reshape(training_x, (training_x.shape[0], training_x.shape[1], training_x.shape[2], 1))
testing_x = tf.reshape(testing_x, (testing_x.shape[0], testing_x.shape[1], testing_x.shape[2], 1))

model = tf.keras.models.Sequential([
    # 1.第一个卷积层
    tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='sigmoid', input_shape=(28, 28, 1)),
    # 2.最大池化
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 3.第二个卷积层
    tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='sigmoid'),
    # 4.最大池化
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 5.打平以便进入全连接
    tf.keras.layers.Flatten(),
    # 6.第一个全连接层
    tf.keras.layers.Dense(512, activation='sigmoid'),
    # 7.第一个全连接层
    tf.keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_x, training_y, epochs=5, validation_split=0.1)
model.evaluate(testing_x, testing_y, verbose=2)
```

