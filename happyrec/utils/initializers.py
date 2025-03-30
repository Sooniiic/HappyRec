import tensorflow as tf
from tensorflow.keras.layers import Layer


class RandomNormal(Layer):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, vocab_size, embed_dim):
        initializer = tf.keras.initializers.RandomNormal(mean=self.mean, stddev=self.std)
        return tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=initializer)

