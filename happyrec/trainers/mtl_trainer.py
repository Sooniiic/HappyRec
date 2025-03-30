import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from happyrec.utils.data import get_loss_func, get_metric_func
from happyrec.utils.mtl import shared_task_layers
from tensorflow.keras.optimizers import Adam
import copy

class MTLTrainer:
    def __init__(
        self,
        model,
        task_types,
        optimizer="Adam",
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        adaptive_params=None,
        n_epoch=10,
        earlystop_taskid=0,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
    ):
        self.model = model
        self.task_types = task_types
        self.n_task = len(task_types)
        
        # 设备配置
        if gpus and len(gpus) > 1:
            strategy = tf.distribute.MirroredStrategy()
            print(f"Running on {strategy.num_replicas_in_sync} GPUs")

        self.loss_fns = [get_loss_func(t) for t in task_types]
        self.evaluate_fns = [get_metric_func(t) for t in task_types]
        
        # 训练配置
        self.n_epoch = n_epoch
        self.earlystop_taskid = earlystop_taskid
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

        self.model.compile(optimizer, loss=['BinaryCrossentropy', 'BinaryCrossentropy'],
                  metrics=['BinaryCrossentropy'])

    def fit(self, x_train, y_train):
        
        for epoch in range(self.n_epoch):
            self.model.fit(x_train,y_train,
                        batch_size=256, epochs=10, verbose=1, validation_split=0.2)
            
            # 验证评估
            # val_metrics = self.evaluate(x_eval, y_eval)
            # print(f"Val metrics: {val_metrics}")
            
        
        # 保存模型
        save_path = os.path.join(self.model_path, f"model")
        self.model.save_weights(save_path)
        return self.early_stopper.best_auc

    def evaluate(self, x_test, y_test):
        task_metrics = [[] for _ in range(self.n_task)]
        y_pred = self.model.predict(x_test, batch_size=256)
        print(y_test.shape)
        print(y_pred.shape)
        for i in range(self.n_task):
            metric = self.evaluate_fns[i](y_test[:, i], y_pred[i])
            task_metrics[i].append(metric.numpy())
        
        return [np.mean(m) for m in task_metrics]

    def predict(self, dataset):
        predictions = []
        for x_batch in dataset:
            y_pred = self.model(x_batch, training=False)
            predictions.append(y_pred.numpy())
        return np.concatenate(predictions)
    

class EarlyStopper:

    def __init__(self, patience):
        self.patience = patience
        self.trial_counter = 0
        self.best_auc = 0
        self.best_weights = None

    def stop_training(self, val_auc, weights):
        """whether to stop training.

        Args:
            val_auc (float): auc score in val data.
            weights (tensor): the weights of model
        """
        if val_auc > self.best_auc:
            self.best_auc = val_auc
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True