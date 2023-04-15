import tensorflow as tf
from tensorflow.keras.layers import Dense, Input


class BaseMLP(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.depth = kwargs["depth"]
        self.width = kwargs["width"]
        self.output_layer = Dense(1, activation="linear")

    def call(self, x):
        x = self.input_layer(x)
        if self.depth:
            for layer in self.hidden_layers:
                x = layer(x)
        elif self.width:
            x = self.hidden_layer(x)
        return self.output_layer(x)
