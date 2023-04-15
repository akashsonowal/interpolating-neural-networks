import pytest
import pandas as pd
from pathlib import Path
import tensorflow as tf
from interpolating_neural_networks.data import FinancialDataset, DistributedDataLoader

def test_folder_contains_files():
    assert all(Path('data').joinpath(file_name).is_file() for file_name in ['c_50.csv', 'r2_50.csv'])

def get_dataset():
    train_datset, val_dataset = FinancialDataset(Path('data'), train_val_split=1/3, input_dim=50, linear=False)()
    return train_dataset, val_dataset 
    
@pytest.fixture
def strategy():
    return tf.distribute.OneDeviceStrategy(device='/cpu:0')

@pytest.fixture
def train_dataset():
    return get_dataset()[0]

@pytest.fixture
def val_dataset():
    return get_dataset()[1]

@pytest.fixture
def dataloader(strategy, train_dataset, val_dataset):
    return DistributedDataLoader(strategy, train_dataset, val_dataset, batch_size=32, num_workers=1)

def test_dataloader_shape(dataloader):
    train_dataloader, val_dataloader = dataloader()
    for x in train_dataloader:
        assert x[0].shape == (32, 100)
        assert x[1].shape == (32, 1)
    for x in val_dataloader:
        assert x[0].shape == (32, 100)
        assert x[1].shape == (32, 1)
