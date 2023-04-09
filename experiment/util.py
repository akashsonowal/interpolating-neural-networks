#!/usr/bin/env python
# coding=utf-8
from typing import Optional, Dict, Any

class WandbCallBack(tf.keras.callbacks.Callback):
  wandb.init(project="inn_smoke_test",
        config={args})

  def __init__(self):
      super(WandbCallBack, self).__init__()
      self.epoch = 0
    
  def on_epoch_end(self, epoch, logs: Optional[Dict[str, Any]] = None) -> None:
      self.epoch += 1
      if logs is None: logs = dict() 
      print("Hi", logs)
      wandb.log(logs)





########################333

from tqdm import tqdm
import tensorflow as tf
from interpolating_neural_network.models import model

if is_wandb_available():
  import wandb

loss_object = tf.keras.losses.MeanSquaredError()


train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')


    
class MLPTrainer(tf.keras.Model):
  
  def __init__(self, model, **kwargs):
    super(MLPTrainer, self).__init__(**kwargs)
    self.model = model
    
  @tf.function
  def train_step(self, features, labels):

    with tf.GradientTape() as tape:
      predictions = model(features, training=True)
      loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(labels, predictions)
  
  @tf.function
  def test_step(self, features, labels):
      predictions = model(features, training=False)
      t_loss = loss_object(labels, predictions)
      test_loss(labels, predictions)
