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
        batch_size =100
        x = tf.random.uniform((batch_size,1), minval=-1, maxval=1)
        f = tf.math.cos(2*x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.math.reduce_mean(tf.math.square(y_pred-f))

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}
    
class PolyTransform(tf.keras.layers.Layer):
    def __init__(self, degree):
        super(PolyTransform, self).__init__()
        self.degree = degree

        # Crear un tensor de potencias de 0 a degree
        self.powers = tf.range(0., self.degree + 1)

        # Crear los pesos para cada término polinómico
        self.coeffs = self.add_weight("coeffs", shape=[self.degree + 1])

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if (inputs.shape == ()):
            inputs=(inputs,)
        elif (len(inputs.shape)==1):
            inputs=tf.expand_dims(inputs, axis=1)
        batch = tf.shape(inputs)[0]
        self.powers_b = tf.ones([batch,1])*self.powers
        terms = inputs ** self.powers_b
        res = tf.tensordot(terms, self.coeffs, 1)
        return tf.expand_dims(res, axis=1)
    
model_F = Funsol()
model_F.add(Dense(100,activation='relu', input_shape=(1,)))
model_F.add(Dense(50,activation='relu'))
model_F.add(Dense(10,activation='relu'))
model_F.add(Dense(1))
model_F.add(PolyTransform(degree=3))
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
plt.plot(x, tf.math.cos(2.*x), label="exact")
plt.legend()
plt.show()
