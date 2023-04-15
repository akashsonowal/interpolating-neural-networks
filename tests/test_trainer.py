import pytest
import tensorflow as tf
from experiment.trainer import MLPDistributedTrainer

@pytest.fixture
def strategy():
    return tf.distribute.OneDeviceStrategy(device="/cpu:0")

@pytest.fixture
def trainer(strategy):
    return MLPDistributedTrainer(strategy, epochs=2, callbacks=None)

@pytest.fixture
def dataset():
    x = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = tf.constant([3, 7, 11, 15, 19])
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
    return dataset

@pytest.fixture
def model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

def test_train_step(trainer, dataset, model):
    global_batch_size = 2
    dataset_iter = iter(dataset)
    dataset_inputs = next(dataset_iter)
    loss = trainer.train_step(dataset_inputs, global_batch_size, model)
    assert loss.numpy() > 0

def test_val_step(trainer, dataset, model):
    dataset_iter = iter(dataset)
    dataset_inputs = next(dataset_iter)
    trainer.val_step(dataset_inputs, model)
    val_loss = trainer.val_loss_metric.result()
    assert val_loss.numpy() >= 0

def test_distributed_train_step(trainer, dataset, model, strategy):
    global_batch_size = 2
    dataset_iter = iter(strategy.experimental_distribute_dataset(dataset))
    dataset_inputs = next(dataset_iter)
    per_replica_losses = trainer.distributed_train_step(dataset_inputs, global_batch_size=global_batch_size, model=model)
    assert per_replica_losses.numpy().shape == (strategy.num_replicas_in_sync,)

def test_distributed_val_step(trainer, dataset, model, strategy):
    dataset_iter = iter(strategy.experimental_distribute_dataset(dataset))
    dataset_inputs = next(dataset_iter)
    trainer.distributed_val_step(dataset_inputs, model=model)

def test_fit(trainer, dataset, model, strategy):
    global_batch_size = 2
    train_dataloader = strategy.experimental_distribute_dataset(dataset)
    val_dataloader = strategy.experimental_distribute_dataset(dataset)
    trainer.fit(model, train_dataloader, val_dataloader, global_batch_size)
    assert trainer.val_loss_metric.result().numpy() >= 0

