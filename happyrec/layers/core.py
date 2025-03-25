"""

@author:

"""
import tensorflow as tf
from tensorflow.keras import Model
from tf.keras.laysers import Layer, Dense, BatchNormalization, Dropout
from happyrec.utils.features import SparseFeature, SequenceFeature, DenseFeature

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

# class EmbeddingLayer(Model):
#     def __init__(self, features):
        
class PredictionLayer(Layer):
    def __init__(self, task_type="classification"):
        super(PredictionLayer, self).__init__()
        if task_type not in ["lassification", "regression"]:
            raise ValueError("task must be 'classfication' or 'regression', {} is illegal".format(task_type))
        self.task_type = task_type

    def forward(self, x):
        if self.task_type == "classification":
            x = tf.sigmoid(x)
        return x
    
class EmbeddingLayer(Layer):
    def __init__(self, features):
        super(EmbeddingLayer, self).__init__()
        self.features = features
        self.embed_dict = {}
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, SequenceFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1

    def call(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []

        for fea in features:
            if isinstance(fea, SparseFeature):
                if fea.shared_with == None:
                    sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1))
                else:
                    sparse_emb.append(self.embed_dict[fea.shared_with](x[fea.name].long()).unsqueeze(1))
                
    


