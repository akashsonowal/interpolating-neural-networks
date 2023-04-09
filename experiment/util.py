#!/usr/bin/env python
# coding=utf-8
import wandb

class WandbCallBack(tf.keras.callbacks.Callback):
  def __init__(self, args):
      wandb.init(project="Interpolating NN Experiments", config={args})
      super(WandbCallBack, self).__init__()
  def on_epoch_end(self, epoch, logs={}):
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
