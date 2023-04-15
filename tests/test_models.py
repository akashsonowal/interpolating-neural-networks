import pytest
import tensorflow as tf
from tensorflow.keras.layers import Dense
from interpolating_neural_networks.models import ExperimentalMLP

@pytest.fixture
def mlp():
  return ExperimentalMLP(input_dim=100, depth=2, width=None)

def test_mlp_output_shape(mlp):
  x = tf.ones((100, mlp.input_dim))
  y = mlp(x)
  assert y.shape == (100, 1)
