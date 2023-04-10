#!/usr/bin/env python
# coding=utf-8
import wandb
import tensorflow as tf

class WandbCallBack(tf.keras.callbacks.Callback):
  def __init__(self, args):
      config_dict = {arg: getattr(args, arg) for arg in vars(args)}
      wandb.init(project="Interpolating NN Experiments", config=config_dict)
      super(WandbCallBack, self).__init__()
  def on_epoch_end(self, epoch, logs={}):
      wandb.log(logs)
