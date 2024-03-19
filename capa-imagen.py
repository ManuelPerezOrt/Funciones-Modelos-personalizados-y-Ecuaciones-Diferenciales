import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image as kimage

class capaGris(keras.layers.Layer):
    def _init_(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        imagenG = tf.image.rgb_to_grayscale(inputs)
        return imagenG
    

# Carga la imagen
img_path = 'train/cat/cat.1.jpg'  # Reemplaza esto con la ruta a tu imagen
img = kimage.load_img(img_path, target_size=(150, 150))  # Cambia el tamaño de la imagen si es necesario

# Convierte la imagen a un array de numpy y agrega una dimensión extra
img_array = kimage.img_to_array(img)
img_array = tf.expand_dims(img_array, axis=0)

# Crea una instancia de la capa y aplica la transformación
gris = capaGris()
img_gris = gris(img_array)
img_gris = tf.squeeze(img_gris)

# Visualiza la imagen en escala de grises
plt.imshow(img_gris, cmap='gray')
plt.show()