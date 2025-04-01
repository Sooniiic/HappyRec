import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from happyrec.layers.core import MLP, PredictionLayer, EmbeddingLayer

class MMOE(Model):
    def __init__(self, features, n_expert, expert_mlp_hidden_units=(16,), tower_mlp_hidden_units=(8,), task_types=['classification', 'classification'], task_names=['ctr_label', 'cvr_label']):
        super(MMOE, self).__init__()
        task_num = len(task_names)
        if task_num <= 1:
            raise ValueError("nums_tasks must be greater than 1")
        if len(task_types) != task_num:
            raise ValueError("num_tasks must be equal to the length of task_types")
        self.features = features
        self.task_types = task_types
        self.embedding = EmbeddingLayer(features)
        self.n_expert = n_expert
        self.n_task = task_num

        self.experts = [MLP(expert_mlp_hidden_units) for i in range (self.n_expert)]
        self.gates = [MLP(hidden_units=(self.n_expert,)) for i in range(self.n_task)]
        self.towers = [MLP(tower_mlp_hidden_units) for i in range(self.n_task)]
        self.predict_layers = [PredictionLayer(task_type, task_name=task_name) for task_type, task_name in zip(task_types, task_names)]

    def call(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)
        expert_outs = [tf.expand_dims(expert(embed_x), axis=1) for expert in self.experts]
        expert_outs = tf.concat(expert_outs, axis=1)
        gate_outs = [tf.expand_dims(gate(embed_x), axis=-1) for gate in self.gates]

        task_outs = []
        for gate_out, tower, predict_layer in zip(gate_outs, self.towers, self.predict_layers):
            expert_weight = tf.multiply(gate_out, expert_outs)
            expert_pooling = tf.reduce_sum(expert_weight, axis=1)
            tower_out = tower(expert_pooling)
            y = predict_layer(tower_out)
            task_outs.append(y)
        return task_outs
    
    def build(self):
        inputs = {
            fea.name: Input(shape=(), dtype=tf.int32, name=fea.name)
            for fea in self.features
        }
        model = Model(inputs=inputs, outputs=self.call(inputs))
        return model

