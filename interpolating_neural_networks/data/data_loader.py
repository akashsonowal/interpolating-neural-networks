import tensorflow as tf

class DistributedDataLoader:
    def __init__(self, strategy, train_dataset, val_dataset, batch_size, num_workers):
        self.strategy = strategy
        self.global_bs = batch_size * num_workers
        self.buffer = len(train_dataset)
        self.train_dataset = self.tensor_slices(train_dataset, train=True)
        self.val_dataset = self.tensor_slices(val_dataset, train=False)
        self.train_dataloader = self.distribute_data(self.train_dataset)
        self.val_dataloader = self.distribute_data(self.val_dataset)
    
    def tensor_slices(self, dataset, train):
        if train:
            return tf.data.Dataset.from_tensor_slices(dataset).shuffle(self.buffer).batch(self.global_bs)
        return tf.data.Dataset.from_tensor_slices(dataset).batch(self.global_bs)
    
    def distribute_data(self, dataset):
        return self.strategy.experimental_distribute_dataset(dataset)
    
    def __call__(self):
        return self.train_dataloader, self.val_dataloader
