import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math

loss_tracker = keras.metrics.Mean(name="loss")
class Funsol(Sequential):
    @property
    def metrics(self):
        return [loss_tracker]

    def train_step(self, data):
        batch_size =100 #Calibra la resolucion
        x = tf.random.uniform((batch_size,1), minval=-1, maxval=1)
        f = 1.+2.*x+4.*x**3

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.math.reduce_mean(tf.math.square(y_pred-f))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}
    
class SinTransform(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(SinTransform,self).__init__()
        self.num_outputs = num_outputs

        self.freq = tf.range(1., self.num_outputs + 1)

        self.kernel = self.add_weight("kernel",
                                shape=[self.num_outputs])
    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if (inputs.shape == ()):
            inputs=(inputs,)
        elif (len(inputs.shape)==1):
            inputs=tf.expand_dims(inputs, axis=1)
        batch = tf.shape(inputs)[0]
        self.freq_b = tf.ones([batch,1])*self.freq
        args = self.freq_b * inputs
        modes = tf.math.sin(args)
        res = tf.tensordot(modes,self.kernel,1)
        return tf.expand_dims(res, axis=1)

model_F = Funsol()
model_F.add(Dense(100,activation='relu', input_shape=(1,)))
model_F.add(Dense(50,activation='relu'))
model_F.add(Dense(10,activation='relu'))
model_F.add(Dense(1))
model_F.add(SinTransform(100))
model_F.build(input_shape=(1,))
model_F.summary()

model_F.compile(optimizer=Adam(learning_rate=0.001), metrics=['loss'])
x=tf.linspace(-1,1,100)
history = model_F.fit(x,epochs=10000,verbose=0)
print(model_F.layers[0].trainable_weights)
plt.plot(history.history["loss"])
plt.show()

a=model_F.predict(x)
plt.plot(x,a,label="aprox")
plt.plot(x, 1.+2.*x+4.*x**3, label="exact")
plt.legend()
plt.show()