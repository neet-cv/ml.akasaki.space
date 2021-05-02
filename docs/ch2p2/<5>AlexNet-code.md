# AlexNet代码实现

## 使用tf.keras高层API的快捷写法（带注释42行）

```python
import tensorflow as tf
from tensorflow.keras import layers, datasets, Sequential

(training_x, training_y), (testing_x, testing_y) = datasets.cifar10.load_data()
training_x = (training_x.astype('float32') / 255.)
testing_x = (testing_x.astype('float32') / 255.)

model = Sequential([
    # Conv block #1
    layers.Conv2D(filters=96, kernel_size=(3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=2),
    # Conv block #2
    layers.Conv2D(filters=256, kernel_size=(3, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=2),
    # Conv block #3
    layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                  activation='relu'),
    # Conv block #4
    layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same',
                  activation='relu'),
    # Conv block #5
    layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same',
                  activation='relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=2),
    # Dense fully connected
    layers.Flatten(),
    layers.Dense(2048, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2048, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(training_x, training_y, batch_size=32, epochs=20, validation_data=(testing_x, testing_y))
```

