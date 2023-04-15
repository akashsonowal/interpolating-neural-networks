#!/usr/bin/env python
import tensorflow as tf
import wandb


class WandbCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_end(self, logs={}):
        wandb.log(logs)
