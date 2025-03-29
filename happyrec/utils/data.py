import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error

def get_auto_embedding_dim(num_classes):
    '''
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    why?
    '''
    return int(np.floor(6 * np.pow(num_classes, 0.25)))


class DataGenerator(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(x)

    def generate_dataloader(self, x_val=None, y_val=None, x_test=None, y_test=None, split_ratio=None, batch_size=16):
        if split_ratio is not None:
            train_size = int(self.length * split_ratio[0])
            val_size = int(self.length * split_ratio[1])
            test_size = self.length - train_size - val_size
            print("The samples of train : val : test are %d : %d : %d" % (train_size, val_size, test_size))
            
            # Split the data into train, val, test
            train_x, val_x, test_x = np.split(self.x, [train_size, train_size + val_size])
            train_y, val_y, test_y = np.split(self.y, [train_size, train_size + val_size])
        else:
            train_x, train_y = self.x, self.y
            val_x, val_y = x_val, y_val
            test_x, test_y = x_test, y_test

        # Create TensorFlow Dataset objects
        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
        test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

        # Shuffle, batch and repeat the training dataset
        train_dataloader = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        # Validation and Test datasets (no shuffle, no repeat)
        val_dataloader = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataloader = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return train_dataloader, val_dataloader, test_dataloader

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
