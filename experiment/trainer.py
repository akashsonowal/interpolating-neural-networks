#!/usr/bin/env python
# in trainer.py
with strategy.scope():
  loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

  def compute_loss(labels, predictions, model_losses):
    per_example_loss = loss_object(labels, predictions) 
    loss = tf.nn.compute_average_loss(per_example_loss,
                                      global_batch_size=GLOBAL_BATCH_SIZE)
    if model_losses:
      loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
    return loss
  
  train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
  test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')

  model = model = ExperimentalMLP(args)
  optimizer = tf.keras.optimizers.Adam()
  
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


class MLPDistributedTrainer:

  def __init__(self, epochs):
    self.epochs = epochs 

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

  def train(self, callbacks):
    for epoch in range(self.epochs):
      train_loss = self.distributed_train_epoch(train_dist_dataset)

      template = ("Epoch {}, Loss: {}")
      print(template.format(epoch + 1, train_loss))

      if callbacks is not None:
        for callback in callbacks:
          print(train_loss.numpy())
          callback.on_epoch_end(epoch, logs={"train_loss": train_loss.numpy()})

      #  # Logging with W&B
      # wandb.log({"train_loss": loss.numpy(),
      #             "val_loss": val_loss.numpy()
      # })
