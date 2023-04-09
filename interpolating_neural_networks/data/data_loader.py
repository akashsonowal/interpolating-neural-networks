x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = args.train_test_split, shuffle=False) 

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = len(x_train)
BATCH_SIZE_PER_REPLICA = args.BATCH_SIZE_PER_REPLICA
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
EPOCHS = args.EPOCHS

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE) 
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(GLOBAL_BATCH_SIZE) 
train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

train_dataloader, val_dataloader =  DistributedDataLoder(train_dataset, val_dataset, 
                                                           batch_size=args.BATCH_SIZE_PER_REPLICA, 
                                                           num_workers=strategy.num_replicas_in_sync)