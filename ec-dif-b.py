import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from matplotlib import pyplot as plt
import numpy as np

class ODEsolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
      return [self.loss_tracker]

    def train_step(self, data):
         batch_size = tf.shape(data)[0]
         min = tf.cast(tf.reduce_min(data),tf.float32)
         max = tf.cast(tf.reduce_max(data),tf.float32)
         x = tf.random.uniform((batch_size,1), minval=min, maxval=max)

         with tf.GradientTape(persistent=True) as tape:
             tape.watch(x)
             with tf.GradientTape() as tape2:
                 tape2.watch(x)
                 y_pred = self(x, training=True)
                 dy = tape2.gradient(y_pred, x) 
             ddy = tape.gradient(dy, x) 
             x_o = tf.zeros((batch_size,1))
             y_o = self(x_o,training=True) 
             dy_o = tape.gradient(y_o, x_o)
             if dy_o is None:
                 dy_o = tf.zeros_like(y_o)
             eq = ddy + y_pred  
             ic = 1. 
             ic_dy = -0.5 
             loss = self.mse(0., eq) + self.mse(y_o,ic) + self.mse(dy_o,ic_dy) 

        # Apply grads
         grads = tape.gradient(loss, self.trainable_variables)
         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
         self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
         return {"loss": self.loss_tracker.result()}
    
model = ODEsolver()
model.add(Dense(100, activation='tanh', input_shape=(1,)))
model.add(Dense(10,activation='tanh'))
model.add(Dense(1))
model.summary()

model.compile(optimizer=RMSprop(learning_rate= 0.001), metrics=['loss'])

x=tf.linspace(-5,5,100)
history = model.fit(x,epochs=3000,verbose=0)
plt.plot(history.history["loss"])
plt.show()

x_testv = tf.linspace(-5,5,100)
a=model.predict(x_testv)
plt.plot(x_testv,a,label="Numérica")
plt.plot(x_testv,np.cos(x)-(np.sin(x))/2,label="Analítica")
plt.legend()
plt.show()