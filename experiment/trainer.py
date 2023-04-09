#!/usr/bin/env python
# trainer = MLPDistributedTrainer(epochs=args.epochs, callbacks=[WandbCallBack()])
# trainer.fit(model, train_dataloader, val_dataloader)
import tensorflow as tf
from .util import strategy, compute_loss, loss_object, train_loss, optimizer
startegy = 

class MLPDistributedTrainer:
  def __init__(self, model, epochs):
    self.model = model
    self.epochs = epochs 
    
  @tf.function
  def train_step(self, inputs):
    features, labels = inputs
    with tf.GradientTape() as tape:
      predictions = self.model(features, training=True)
      loss = compute_loss(labels, predictions, model.losses)

    gradients = tape.gradient(loss, self.model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    train_loss.update_state(labels, predictions)
    return loss 
  
  @tf.function
  def test_step(self, inputs):
    features, labels = inputs
    predictions = self.model(features, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss.update_state(labels, predictions) #(t_loss)

  @tf.function
  def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                           axis=None)

  @tf.function
  def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))

  def fit(self, model, train_dataloader, val_dataloader):
      with strategy.scope():
        val_loss = tf.keras.metrics.Mean(name='val_loss')
      
      for epoch in range(self.epochs):
        # Train Loop
        total_loss = 0.0
        num_batches = 0
        for x in train_data_loader:
          total_loss += self.distributed_train_step(model, x)
          num_batches += 1
        train_loss = total_loss / num_batches
        # Validation Loop
        for x in val_data_loader:
          self.distributed_test_step(model, x)
        val_loss.reset_states()

      if self.callbacks is not None:on
        for callback in self.callbacks:
          callback.on_train_end(logs={"train_loss": train_loss.numpy(), "val_loss": val_loss.numpy()})
