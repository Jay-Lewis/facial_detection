import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_train_batch_begin(self, batch, logs={}):
        pass
    def on_train_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

history = LossHistory()
model.fit(x_train, y_train, epochs=1, callbacks=[history])
model.evaluate(x_test,  y_test, verbose=2)



plt.figure()
xs = [x for x in range(0,np.size(history.losses))]
plt.plot(xs, history.losses)
plt.title("Loss vs Training Steps")
plt.figure()

xs = [x for x in range(0,np.size(history.accuracies))]
plt.plot(xs, history.accuracies)
plt.title("Accuracy vs Training Steps")
plt.show()
