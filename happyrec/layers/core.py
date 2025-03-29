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
    
class SumPooling(Layer):
    def __init__(self):
        super(SumPooling, self).__init__()
    def call(self, x, mask=None):
        if mask is None:
            return tf.reduce_sum(x, axis=1)
        else:
            return tf.squeeze(tf.linalg.matmul(mask, x), axis=1)

class AveragePooling(Layer):
    def __init__(self):
        super(AveragePooling, self).__init__()
    def call(self, x, mask=None):
        if mask is None:
            return tf.reduce_mean(x, axis=1)
        else:
            sum_pooling_matrix = tf.squeeze(tf.linalg.matmul(mask, x), axis=1)
            non_padding_length = tf.reduce_sum(mask, axis=-1)
            return sum_pooling_matrix / (tf.cast(non_padding_length, tf.float32) + 1e-16)

class InputMask(Layer):
    def __init__(self):
        super(InputMask, self).__init__()

    def call(self, x, features):
        mask = []
        if not isinstance(features, list):
            features = [features]
        
        for fea in features:
            if isinstance(fea, SparseFeature) or isinstance(fea, SequenceFeature):
                if fea.padding_idx is not None:
                    fea_mask = x[fea.name].long() != fea.padding_idx
                else:
                    fea_mask = x[fea.name].long() != -1
                mask.append(tf.cast(tf.expand_dims(fea_mask, axis=1), tf.float32))
            else:
                raise ValueError("Only SparseFeature or SequenceFeature support to get mask.")
        return tf.concat(mask, axis=1)
        

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
        sparse_exists, dense_exists = False, False

        for fea in features:
            if isinstance(fea, SparseFeature):
                if fea.shared_with == None:
                    sparse_emb.append(tf.expand_dims(self.embed_dict[fea.name](x[fea.name].long()), axis=1))
                else:
                    sparse_emb.append(tf.expand_dims(self.embed_dict[fea.shared_with](x[fea.name].long()), axis=1))
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = SumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = AveragePooling()
                else:
                    raise ValueError("Sequence pooling method supports only {'sum', 'mean'}, but got {}.".format(fea.pooling))
                fea_mask = InputMask()(x, fea)
                if fea.shared_with == None:
                    sparse_emb.append(tf.expand_dims(pooling_layer(self.embed_dict[fea.name](x[fea.name].long(), fea_mask)), axis=1))
                else:
                    sparse_emb.append(tf.expand_dims(pooling_layer(self.embed_dict[fea.shared_with](x[fea.name].long()), fea_mask), axis=1))
            else:
                dense_values.append(tf.cast(x[fea.name], tf.float32) if len(x[fea.name].shape) > 1 else tf.expand_dims(tf.cast(x[fea.name], tf.float32), axis=1))

        if len(dense_values) > 0:
            dense_exists = True
            dense_values = tf.concat(dense_values, axis=1)
        if len(sparse_emb) > 0:
            sparse_exists = True
            sparse_emb = tf.concat(sparse_emb, axis=2)
        
        if squeeze_dim:
            if dense_exists and not sparse_exists:
                return dense_values
            elif not dense_exists and sparse_exists:
                return tf.resahpe(sparse_emb, [tf.shape(sparse_emb)[0], -1])
            elif dense_exists and sparse_exists:
                return tf.concat([tf.reshape(sparse_emb, [tf.shape(sparse_emb)[0], -1]), dense_values], axis=1)
            else:
                raise ValueError("The input features cannot be empty")
        else:
            if sparse_exists:
                return sparse_emb
            else:
                raise ValueError("If keeping the original shape: [batch_size, num_features, embed_dim], expected SparseFeatures in feature list, got {}".format(features))


            

