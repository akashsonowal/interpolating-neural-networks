from tensorflow.keras.layers import Input, Dense

class BaseMLP(tf.keras.Model):
  def __init__(self):
    super(BaseMLP, self).__init__()
    self.output_layer = Dense(1, activation='linear')
  
  def call(self, x):
    x = self.input_layer(x)
    for layer in self.hidden_layers:
      x = layer(x)
    return self.output_layer(x)
