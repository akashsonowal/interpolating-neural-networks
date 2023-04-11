import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

class BaseMLP(tf.keras.Model):
  def __init__(self, strategy, **kwargs):
    super(BaseMLP, self).__init__()
    self.strategy = strategy
    with self.strategy.scope():
      self.depth = kwargs['depth']
      self.width = kwargs['width']
      self.output_layer = Dense(1, activation='linear')
  
  def call(self, x):
    with self.strategy.scope():
      x = self.input_layer(x)
      if self.depth:
        for layer in self.hidden_layers:
          x = layer(x)
      elif self.width:
        x = self.hidden_layer(x)
      return self.output_layer(x)
