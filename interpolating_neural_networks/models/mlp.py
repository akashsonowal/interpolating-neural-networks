import tensorflow as tf
from tensorflow.keras.layers import Dense
from .util import BaseNN

class DeepNN(BaseNN):
  def __init__(self, depth, input_dim):
    super(DeepNN, self).__init__(input_dim)
    self.hidden_layers = [Dense(32, activation='relu') for _ in range(depth)]
  
class WideNN(BaseNN):
  def __init__(self, width, input_dim):
    super(WideNN, self).__init__(input_dim)
    self.hidden_layers = [Dense(width, activation='relu') for _ in range(2)]
