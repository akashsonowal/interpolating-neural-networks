import tensorflow as tf

strategy = tf.distribute.MirroredStrategy()

class DistributedDataLoader:
    def __init__(self, train_dataset, val_dataset, batch_size, num_workers):
        self.global_bs = batch_size * num_workers
        self.train_dataset = self.tensor_slices(train_dataset, True)
        self.val_dataset = self.tensor_slices(val_dataset, False)
        self.train_dataloader = self.distribute_data(self.train_dataset)
        self.val_dataloader = self.distribute_data(self.val_dataset)
    
    def tensor_slices(self, dataset, train):
        if train:
            return tf.data.Dataset.from_tensor_slices(dataset).shuffle(len(train_dataset)).batch(self.global_bs)
        return tf.data.Dataset.from_tensor_slices(dataset).batch(self.global_bs)
    
    def distribute_data(self, dataset):
        return strategy.experimental_distribute_dataset(dataset)
    
    def __call__(self):
        return self.train_dataloader, self.val_dataloader

# BUFFER_SIZE = len(x_train)
# BATCH_SIZE_PER_REPLICA = args.BATCH_SIZE_PER_REPLICA
# GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
# EPOCHS = args.EPOCHS

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE) 
# train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
# test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

# train_dataloader, val_dataloader =  DistributedDataLoder(train_dataset, val_dataset, 
#                                                            batch_size=args.BATCH_SIZE_PER_REPLICA, 
                                                        #    num_workers=strategy.num_replicas_in_sync)