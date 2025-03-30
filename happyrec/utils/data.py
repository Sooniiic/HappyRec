import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error

def get_auto_embedding_dim(num_classes):
    '''
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    why?
    '''
    return int(np.floor(6 * np.pow(num_classes, 0.25)))

def get_loss_func(task_type="classification"):
    if task_type == "classification":
        tf.keras.losses.BinaryCrossentropy()
    elif task_type == "regression":
        tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError("task_type must be classification or regression")


def get_metric_func(task_type="classification"):
    if task_type == "classification":
        return roc_auc_score
    elif task_type == "regression":
        return mean_squared_error
    else:
        raise ValueError("task_type must be classification or regression")
