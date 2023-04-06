import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

class BaseNN(tf.keras.Model):
  def __init__(self, input_dim):
    super(BaseNN, self).__init__()
    self.input_layer = Input(shape=(input_dim, ))
    self.hidden_layers = None
    self.output_layer = Dense(1, activation='linear')
  
  def call(self, x):
    x = self.input_layer(x)
    for layer in self.hidden_layers:
      x = layer(x)
    return self.output_layer(x)


class DeepNN(tf.keras.Model):
  def __init__(self, depth):
    super(DeepNN, self).__init__()
    pass
  
  def call(self, x):
    pass
  
class WideNN(tf.keras.Model):
  def __init__(self, width):
    super(WideNN, self).__init__()
    pass
  
  def call(self, x):
    pass
