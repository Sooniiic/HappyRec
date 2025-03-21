"""

@author:

"""
import tensorflow as tf
from tf.keras.laysers import Layer, Dense, BatchNormalization, Dropout

# 顺序应该是 linear->batch->relu->drop? but 2.0 的dense把激活函数融合进去了，所以先试一下顺序会不会有影响？
class MLP(Layer):
    def __init__(self, hidden_units = None, activation = 'relu', drop_rate = 0, use_batchnorm = True):
        super(MLP, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(drop_rate)
        if hidden_units is None:
            hidden_units = [64, 32, 16]
        self.layers = [Dense(units=unit, activation=activation) for unit in hidden_units]
        
    def call(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        if self.use_batchnorm:
            x = self.batch_norm(x)
        x = self.dropout(x)
        return x

