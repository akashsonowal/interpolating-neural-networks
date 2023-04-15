import pytest
import pandas as pd
from pathlib import Path
import tensorflow as tf
import logging
from interpolating_neural_networks.data import FinancialDataset, DistributedDataLoader

@pytest.mark.parametrize("filepath, exists",
                             [
                                 ("c_50.csv", True),
                                 ("c_200.csv", False),
                              ]
                        )
@pytest.fixture
def data_dir():
    return Path('data')

class TestFinancialDataset:
    def test_data_exists(self, filepath, exists):
        if (data_dir / filepath).exists():
            logging.info(f'{filepath} exists')
        elif not (data_dir / filepath).exists():
            logging.info(f'{filepath} does not exists')

@pytest.fixture
def strategy():
    return tf.distribute.OneDeviceStrategy(device='/cpu:0')

@pytest.fixture
def train_dataset():
    return tf.random.uniform((1000, 10))

@pytest.fixture
def val_dataset():
    return tf.random.uniform((100, 10))

@pytest.fixture
def dataloader(strategy, train_dataset, val_dataset):
    return DistributedDataLoader(strategy, train_dataset, val_dataset, batch_size=20, num_workers=1)

def test_dataloader_shape(dataloader):
    train_dataloader, val_dataloader = dataloader()
    for x in train_dataloader:
        assert x.shape == (20, 10)
    for x in val_dataloader:
        assert x.shape == (20, 10)
