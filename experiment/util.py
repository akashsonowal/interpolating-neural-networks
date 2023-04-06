from tqdm import tqdm
import tensorflow as tf
from interpolating_neural_network.models import model

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')

@tf.function
def train_step(features, labels):
  with tf.GradientTape() as tape:
    predictions = model(features, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  
@tf.function
def train_step(features, labels):
    predictions = model(features, training=False)
    t_loss = loss_object(labels, predictions)
    
    test_loss(t_loss)
  
