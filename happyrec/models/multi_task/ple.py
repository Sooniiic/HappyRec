import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from happyrec.layers.core import MLP, PredictionLayer, EmbeddingLayer

class PLE(Model):
    def __init__(self, features, n_level, n_specific_expert, n_shared_expert, expert_mlp_hidden_units=(16,), tower_mlp_hidden_units=(8,), task_types=['classification', 'classification'], task_names=['ctr_label', 'cvr_label']):
        super(PLE, self).__init__()
        task_num = len(task_names)
        if task_num <= 1:
            raise ValueError("nums_tasks must be greater than 1")
        if len(task_types) != task_num:
            raise ValueError("num_tasks must be equal to the length of task_types")
        self.features = features
        self.task_types = task_types
        self.embedding = EmbeddingLayer(features)
        self.cgc_layers = [CGC(i+1, n_level, task_num, n_specific_expert, n_shared_expert, expert_mlp_hidden_units) for i in range(n_level)]
        self.towers = [MLP(tower_mlp_hidden_units) for i in range(task_num)]
        self.predict_layers = [PredictionLayer(task_type, task_name=task_name) for task_type, task_name in zip(task_types, task_names)]

        self.n_level = n_level
        self.n_task = task_num
    def call(self, x):
        embed = self.embedding(x, self.features, squeeze_dim=True)
        ple_inputs = [embed] * (self.n_task + 1)
        ple_outs = []
        for i in range(self.n_level):
            ple_outs = self.cgc_layers[i](ple_inputs)
            ple_inputs = ple_outs
        
        task_outs = []
        for ple_out, tower, predict_layer in zip(ple_outs, self.towers, self.predict_layers):
            tower_out = tower(ple_out)
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



class CGC(Model):
    def __init__(self, cur_level, n_level, n_task, n_specific_expert, n_shared_expert, expert_mlp_hidden_units=(16,)):
        super(CGC, self).__init__()
        self.cur_level = cur_level
        self.n_level = n_level
        self.n_task = n_task
        self.n_specific_expert = n_specific_expert
        self.n_shared_expert = n_shared_expert
        self.n_all_expert = n_specific_expert * n_task + n_shared_expert

        self.specific_experts = [MLP(expert_mlp_hidden_units) for _ in range(self.n_task * self.n_specific_expert)]
        self.shared_experts = [MLP(expert_mlp_hidden_units) for _ in range(self.n_shared_expert)]
        self.specific_gates = [MLP((self.n_specific_expert+self.n_shared_expert,), activation='softmax') for _ in range (self.n_task)]

        if cur_level < n_level:
            self.shared_gate = MLP((self.n_all_expert, ), activation='softmax')
        
    def call(self, x):
        specific_expert_outs = []
        for i in range(self.n_task):
            specific_expert_outs.extend([
                tf.expand_dims(expert(x[i]), axis=1)
                for expert in self.specific_experts[i * self.n_specific_expert : (i+1) * self.n_specific_expert]
            ])
        shared_expert_outs = [tf.expand_dims(expert(x[-1]), axis=1) for expert in self.shared_experts]
        specific_gate_outs = [tf.expand_dims(gate(x[i]), axis=-1) for i, gate in enumerate(self.specific_gates)]

        cgc_outs = []
        for i, gate_out in enumerate(specific_gate_outs):
            cur_expert_list = specific_expert_outs[i * self.n_specific_expert : (i+1) * self.n_specific_expert] + shared_expert_outs
            expert_concat = tf.concat(cur_expert_list, axis=1)
            expert_weight = tf.multiply(gate_out, expert_concat)
            expert_pooling = tf.reduce_sum(expert_weight, axis=1)
            cgc_outs.append(expert_pooling)
        if self.cur_level < self.n_level:
            shared_gate_out = self.shared_gate(tf.expand_dims(x[-1], axis=-1))
            expert_concat = tf.concat(specific_expert_outs + shared_expert_outs, axis=1)
            expert_weight = tf.multiply(shared_gate_out, expert_concat)
            expert_pooling = tf.reduce_sum(expert_weight, axis=1)
            cgc_outs.append(expert_pooling)
        return cgc_outs



