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

class DeepNN(BaseNN):
  def __init__(self, depth, input_dim):
    super(DeepNN, self).__init__(input_dim)
    self.hidden_layers = [Dense(32, activation='relu') for _ in range(depth)]
  
class WideNN(BaseNN):
  def __init__(self, width, input_dim):
    super(WideNN, self).__init__(input_dim)
    self.hidden_layers = [Dense(width, activation='relu') for _ in range(2)]