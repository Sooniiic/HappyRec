import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from happyrec.utils.data import get_loss_func, get_metric_func
from happyrec.utils.mtl import shared_task_layers
import copy

class MTLTrainer:
    def __init__(
        self,
        model,
        task_types,
        optimizer_fn=tf.keras.optimizers.Adam,
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
            with strategy.scope():
                self._initialize_components(optimizer_fn, optimizer_params, adaptive_params)
        else:
            self._initialize_components(optimizer_fn, optimizer_params, adaptive_params)
        
        # 损失函数和评估指标
        self.loss_fns = [get_loss_func(t) for t in task_types]
        self.evaluate_fns = [get_metric_func(t) for t in task_types]
        
        # 训练配置
        self.n_epoch = n_epoch
        self.earlystop_taskid = earlystop_taskid
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def _initialize_components(self, optimizer_fn, optimizer_params, adaptive_params):
        """初始化优化器、自适应组件等"""
        # 优化器参数默认值
        if optimizer_params is None:
            optimizer_params = {"learning_rate": 1e-3, "weight_decay": 1e-5}
        
        # 默认优化器
        if self.adaptive_method != "metabalance":
            self.optimizer = optimizer_fn(**optimizer_params)

    def train_one_epoch(self, train_dataset):
        total_loss = np.zeros(self.n_task)
        progbar = tqdm(train_dataset, desc="Training")
        
        for step, (x_dict, ys) in enumerate(progbar):
            # 转换数据格式
            ys = [ys[:, i] for i in range(self.n_task)]
            
            # 计算损失和梯度
            with tf.GradientTape(persistent=True) as tape:
                y_preds = self.model(x_dict, training=True)
                loss_list = []
                for i in range(self.n_task):
                    # if isinstance(self.model, ESMM) and i == 0:
                    #     continue  # 跳过ESMM的CTR任务
                    loss = self.loss_fns[i](ys[i], y_preds[i])
                    loss_list.append(loss)
                total_loss = tf.add_n(loss_list) / len(loss_list)
                
                # # 自适应损失加权
                # if self.adaptive_method == "uwl":
                #     total_loss = 0.0
                #     for i, (loss, w) in enumerate(zip(loss_list, self.loss_weight)):
                #         w_clipped = tf.maximum(w, 0.0)
                #         total_loss += 2 * loss * tf.exp(-w_clipped) + w_clipped
                # else:
                    
            
            # # 梯度计算和应用
            # if self.adaptive_method == "metabalance":
            #     # 元平衡特殊处理
            #     share_vars = self.model.shared_layers.trainable_variables
            #     task_vars = self.model.task_specific_layers.trainable_variables
                
            #     # 共享层梯度
            #     share_grads = tape.gradient(total_loss, share_vars)
            #     self.share_optimizer.apply_gradients(zip(share_grads, share_vars))
                
            #     # 任务层梯度
            #     task_grads = tape.gradient(total_loss, task_vars)
            #     self.task_optimizer.apply_gradients(zip(task_grads, task_vars))
                
            #     # 元优化步骤
            #     self.meta_optimizer.step(loss_list)
            # elif self.adaptive_method == "gradnorm":
            #     # 梯度标准化处理
            #     if self.initial_task_loss is None:
            #         self.initial_task_loss = [l.numpy() for l in loss_list]
                
            #     # 计算梯度标准
            #     with tape.stop_recording():
            #         grads = tape.gradient(loss_list, self.last_share_layer.kernel)
            #         gradnorm(loss_list, self.loss_weight, grads, self.initial_task_loss, self.alpha)
                
            #     # 应用梯度
            #     vars = self.model.trainable_variables + self.loss_weight
            #     grads = tape.gradient(total_loss, vars)
            #     self.optimizer.apply_gradients(zip(grads, vars))
                
            #     # 权重归一化
            #     total_weight = sum([w.numpy() for w in self.loss_weight])
            #     for w in self.loss_weight:
            #         w.assign(w * len(self.loss_weight) / total_weight)
            # else:
            #     # 常规优化
                vars = self.model.trainable_variables
                # if self.adaptive_method == "uwl":
                #     vars += self.loss_weight
                grads = tape.gradient(total_loss, vars)
                self.optimizer.apply_gradients(zip(grads, vars))
            
            # 记录损失
            for i in range(self.n_task):
                total_loss[i] += loss_list[i].numpy()
            
            progbar.set_postfix({f'task_{i}_loss': total_loss[i]/(step+1) for i in range(self.n_task)})
        
        return [total_loss[i]/(step+1) for i in range(self.n_task)]

    def fit(self, train_dataset, val_dataset, mode='base', seed=0):
        best_weights = None
        
        for epoch in range(self.n_epoch):
            print(f"\nEpoch {epoch+1}/{self.n_epoch}")
            train_loss = self.train_one_epoch(train_dataset)
            
            # 验证评估
            val_metrics = self.evaluate(val_dataset)
            print(f"Val metrics: {val_metrics}")
            
            # 早停判断
            if self.early_stopper.stop_training(val_metrics[self.earlystop_taskid], self.model.get_weights()):
                print(f'Early stopping at epoch {epoch}')
                self.model.set_weights(self.early_stopper.best_weights)
                break
        
        # 保存模型
        save_path = os.path.join(self.model_path, f"model_{mode}_{seed}")
        self.model.save_weights(save_path)
        return self.early_stopper.best_auc

    def evaluate(self, dataset):
        task_metrics = [[] for _ in range(self.n_task)]
        
        for x_batch, y_batch in dataset:
            y_pred = self.model(x_batch, training=False)
            for i in range(self.n_task):
                metric = self.evaluate_fns[i](y_batch[:, i], y_pred[i])
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