#!/usr/bin/env python
# coding=utf-8
# import wandb
import tensorflow as tf

class WandbCallBack(tf.keras.callbacks.Callback):
  def __init__(self):
      super(WandbCallBack, self).__init__()
  def on_train_end(self, logs={}):
      print(logs)
#       wandb.log(logs)
