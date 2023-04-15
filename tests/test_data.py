import pytest
import tensorflow as tf
from interpolating_neural_networks.data import DistributedDataLoader

@pytest.fixture
def strategy():
    return tf.distribute.OneDeviceStrategy(device='/cpu:0')

@pytest.fixture
def train_dataset():
    return tf.data.Dataset.from_tensor_slices(tf.random.uniform((1000, 10)))

@pytest.fixture
def val_dataset():
    return tf.data.Dataset.from_tensor_slices(tf.random.uniform((100, 10)))

@pytest.fixture
def dataloader(strategy, train_dataset, val_dataset):
    return DistributedDataLoader(strategy, train_dataset, val_dataset, batch_size=10, num_workers=1)

def test_dataloader_shape(dataloader):
    train_dataloader, val_dataloader = dataloader()
    for x in train_dataloader:
        assert x.shape == (10, 10)
    for x in val_dataloader:
        assert x.shape == (10, 10)
