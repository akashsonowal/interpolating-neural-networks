#!/usr/bin/env python
# trainer = MLPDistributedTrainer(epochs=args.epochs, callbacks=[WandbCallBack()])
# trainer.fit(model, train_dataloader, val_dataloader)
import tensorflow as tf
from .util import strategy, compute_loss, train_loss, optimizer

class MLPDistributedTrainer:
  def __init__(self, model, epochs):
    self.model = model
    self.epochs = epochs 
  
  @tf.function
  def train_step(self, inputs):
    features, labels = inputs
    with tf.GradientTape() as tape:
      predictions = self.model(features, training=True)
      loss = compute_loss(features, predictions, model.losses)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    train_loss.update_state(labels, predictions)
    return loss 
    

  @tf.function
  def distributed_train_epoch(self, dataset):
    total_loss = 0.0
    num_batches = 0
    for x in dataset:
      per_replica_losses = strategy.run(train_step, args=(x,))
      total_loss += strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
      num_batches += 1
    return total_loss / tf.cast(num_batches, dtype=tf.float32)

  def fit(self, model, train_dataloader, val_dataloader):
    for epoch in range(self.epochs):
      train_loss = self.distributed_train_epoch(train_dist_dataset)

      template = ("Epoch {}, Loss: {}")
      print(template.format(epoch + 1, train_loss))

      if callbacks is not None:
        for callback in callbacks:
          print(train_loss.numpy())
          callback.on_epoch_end(epoch, logs={"train_loss": train_loss.numpy()})


################3

  
# logger.info('Building model....')

with strategy.scope():
  model = DeepNN(5, 10)
  optimizer = tf.keras.optimizers.Adam()
  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
  
def train_step(inputs):
  features, labels = inputs

  with tf.GradientTape() as tape:
    predictions = model(features, training=True)
    loss = compute_loss(features, predictions, model.losses)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss.update_state(labels, predictions)
  return loss 

def test_step(inputs):
  features, labels = inputs

  predictions = model(features, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss.update_state(labels, predictions)

@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
  return strategy.run(test_step, args=(dataset_inputs,))
