import pytest
import tensorflow as tf
from interpolating_neural_networks.models import ExperimentalMLP

@pytest.fixture
def mlp():
  return ExperimentalMLP(input_dim=100, depth=2, width=None)

def test_mlp_output_shape(mlp):
  x = tf.ones((100, 100))
  y = mlp(x)
  assert y.shape == (100, 1)
