#!/usr/bin/env python
# coding=utf-8
import wandb
import tensorflow as tf

class WandbCallBack(tf.keras.callbacks.Callback):
  def __init__(self, args):
      wandb.init(project="Interpolating NN Experiments", config=vars(args)
      super(WandbCallBack, self).__init__()
  def on_epoch_end(self, epoch, logs={}):
      wandb.log(logs)
