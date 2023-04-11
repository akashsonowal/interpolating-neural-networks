import tensorflow as tf
from tensorflow.keras.layers import Dense
from .util import BaseMLP

# model = ExperimentalMLP(input_dim=args.input_dim, depth=depth, width=None)
    
class ExperimentalMLP(BaseMLP):
  def __init__(self, strategy, **kwargs):
    super(ExperimentalMLP, self).__init__(strategy, **kwargs)
    self.strategy = strategy
    
    with self.strategy.scope():
        if kwargs['depth'] is not None:
          self.input_layer = Dense(32, input_shape=(kwargs['input_dim'], ))
          self.hidden_layers = [Dense(32, activation='relu') for _ in range(kwargs['depth']-1)]
        elif kwargs['width'] is not None: 
          self.input_layer = Dense(kwargs['width'], input_shape=(kwargs['input_dim'], ))
          self.hidden_layer = Dense(kwargs['width'], activation='relu')
