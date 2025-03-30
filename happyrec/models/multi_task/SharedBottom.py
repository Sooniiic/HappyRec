"""
Date:
Reference:
Author:
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from happyrec.layers.core import MLP, PredictionLayer, EmbeddingLayer

class SharedBottom(Model):
    def __init__(self, features, bottem_mlp_hidden_units=(256, 128), tower_mlp_hidden_units=(64, 1), 
                l2_reg_embedding=0.00001, l2_reg_dnn=0, drop_rate=0, activation='relu',
                dnn_use_bn=True, task_types=['classification', 'regression'], task_names=['ctr', 'cvr']):
        super(SharedBottom, self).__init__()
        num_tasks = len(task_names)
        if num_tasks <= 1:
            raise ValueError("num_tasks must be greater than 1")
        if len(task_types) != num_tasks:
            raise ValueError("num_tasks must be equal to the length og task_types")

        self.features = features
        self.task_types = task_types
        self.embedding = EmbeddingLayer(features)

        self.bottom_mlp = MLP(bottem_mlp_hidden_units, activation, drop_rate)
        self.towers = [MLP(tower_mlp_hidden_units, activation, drop_rate) for task_type in task_types]
        self.predict_layers = [PredictionLayer(task_type) for task_type in task_types]

    def call(self, x):
        x = self.embedding(x, self.features, squeeze_dim=True)
        print(x.shape)
        x = self.bottom_mlp(x)
        outputs = [predict_layer(tower(x)) for tower, predict_layer in zip(self.towers, self.predict_layers)]
        return tf.concat(outputs, axis=-1)
    
    def summary(self):
        inputs = {
            fea.name: Input(shape=(), dtype=tf.int32, name=fea.name)
            for fea in self.features
        }
        Model(inputs=inputs, outputs=self.call(inputs)).summary()




        
        
