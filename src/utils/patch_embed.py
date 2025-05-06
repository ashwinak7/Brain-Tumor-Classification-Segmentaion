import tensorflow as tf
from tensorflow.keras import layers

class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, patch_size=16, embed_dim=256, **kwargs):
        super().__init__(**kwargs)  # <-- Important: pass **kwargs to base class
        self.patch_size = patch_size
        self.projection = tf.keras.layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)

    def call(self, x):
        x = self.projection(x)
        x = tf.reshape(x, (tf.shape(x)[0], -1, tf.shape(x)[-1]))
        return x
