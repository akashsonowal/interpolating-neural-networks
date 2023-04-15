import pytest
import pandas as pd
from pathlib import Path
import tensorflow as tf
from interpolating_neural_networks.data import FinancialDataset, DistributedDataLoader

def test_folder_contains_files():
    assert all(Path('data').joinpath(file_name).is_file() for file_name in ['c_50.csv', 'r2_50.csv'])

def test_dataset():
    dataset = FinancialDataset(Path('data'), train_val_split=1/3, input_dim=50, linear=False)
    assert isinstance(dataset, FinancialDataset)
    
@pytest.fixture
def strategy():
    return tf.distribute.OneDeviceStrategy(device='/cpu:0')

@pytest.fixture
def train_dataset():
    return (tf.random.uniform((1000, 100)), tf.random.uniform((1000, 1)))

@pytest.fixture
def val_dataset():
    return (tf.random.uniform((100, 100)), tf.random.uniform((100, 1)))

@pytest.fixture
def dataloader(strategy, train_dataset, val_dataset):
    return DistributedDataLoader(strategy, train_dataset, val_dataset, batch_size=20, num_workers=1)

def test_dataloader_shape(dataloader):
    train_dataloader, val_dataloader = dataloader()
    for x in train_dataloader:
        assert x[0].shape == (20, 100)
        assert x[1].shape == (20, 1)
    for x in val_dataloader:
        assert x[0].shape == (20, 100)
        assert x[1].shape == (20, 1)
