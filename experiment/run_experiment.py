import argparse
from pathlib import Path

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

trainer = MLPTrainer.compile(optimizer=optimizer, loss="mse")

if is_wandb_available():
  import wandb

logger.info('Say Hi')

np.random_seed(42)
tf.random_set_seed(42)

def _setup_parser():
  """Setup Python's ArgumentParser with data, model, trainer, and other arguments."""
  pass

def main():
  parser = _setup_parser()
  args = parser.parse_args()
  data, model = setup_data_and_model_from_args(args)
  
  log_dir = Path("training") / "logs"
  _ensure_logging_dir(log_dir)
  
  if args.wandb:
    logger = 
    
  trainer = xyz
  trainer.fit(model, datamodule)

if __name__ == "__main__":
  main()



class TrainNN:
  pass

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

loss_object = tf.keras.losses()
optimizer = tf.keras.optimizers.Adam()
train_metric = tf.keras.metrics.Mean(name="Train Loss")
test_metric = tf.keras.metrics.Mean(name="Train Loss")
  
EPOCHS = 5

with mirrored_strategy.scope():
  model = xyz
  optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.BinaryCrossentropy(
  from_logits=True,
  reduction=tf.keras.losses.Reduction.NONE)

def compute_loss(labels, predictions):
  per_example_loss = loss_object(labels, predictions)
  return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

def train_step(inputs):
  features, labels = inputs

  with tf.GradientTape() as tape:
    predictions = model(features, training=True)
    loss = compute_loss(labels, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss

@tf.function
def distributed_train_step(dist_inputs):
  per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
  return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )
  
  

class NewRiskCurveExperiment:
  def __init__(self):
    pass
  
  def run():
    
    
    
    with tf.GradientTape() as tape:
      
    pass
  
  def plot():
    pass

if __name__ == '__main__':
  exp = NewRiskCurveExperiment()
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  exp.run()
