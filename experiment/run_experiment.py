import argparse
from pathlib import Path

import random
import numpy as np
import tensorflow as tf

from experiment.util import WandbCallBack
from interpolating_neural_networks.data import FinancialDataset, DistributedDataLoader
from interpolating_neural_networks.models import ExperimentalMLP
from experiment.trainer import MLPDistributedTrainer
strategy = tf.distribute.MirroredStrategy()

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def get_args_parser():
    parser = argparse.ArgumentParser('Interpolating NN experiment setup')
    parser.add_argument('--train_val_split', default=1/3, type=float, help='train val split ratio')
    parser.add_argument('--batch_size_per_replica', default=32, type=int, help='batch size of training on a single GPU')
    parser.add_argument('--epochs', default=15, type=int, help='training epochs')
    parser.add_argument('--input_dim', default=100, type=int, help='input feature set size')
    parser.add_argument('--linear', default=False, const=False, nargs='?', choices=[False, True], help='linear pattern in data')
    parser.add_argument('--expt_type', default='depth', const='depth', nargs='?', choices=['depth', 'width'], help='experiment type')
    parser.add_argument('--depths', default=[5, 10, 15], type=int, nargs='+', help='depth of NNs for increasing depth experiment (list of integers)')
    parser.add_argument('--widths', default=[16, 32, 64], type=int, nargs='+', help='width of NNs for increasing width experiment (list of integers)')
    return parser.parse_args()

def main(args):
  data_dir = Path("data/")
  strategy = tf.distribute.MirroredStrategy()
  print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

  train_dataset, val_dataset = FinancialDataset(data_dir, 
                                                train_val_split=args.train_val_split,
                                                input_dim=args.input_dim, 
                                                linear=args.linear)()

  train_dataloader, val_dataloader =  DistributedDataLoader(train_dataset, 
                                                           val_dataset, 
                                                           batch_size=args.batch_size_per_replica, 
                                                           num_workers=strategy.num_replicas_in_sync)()

  with strategy.scope():
    loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    wandb_callbacks = WandbCallBack(args)
    if args.expt_type=='depth':
      for depth in args.depths:
          model = ExperimentalMLP(input_dim=args.input_dim, depth=depth, width=None)
          trainer = MLPDistributedTrainer(epochs=args.epochs, callbacks=[wandb_callbacks])
          trainer.fit(model, train_dataloader, val_dataloader, args.batch_size_per_replica * strategy.num_replicas_in_sync)
    else:
      for width in args.widths:
          model = ExperimentalMLP(input_dim=args.input_dim, depth=None, width=width)
          trainer = MLPDistributedTrainer(epochs=args.epochs, callbacks=[wandb_callbacks])
          trainer.fit(model, train_dataloader, val_dataloader, args.batch_size_per_replica * strategy.num_replicas_in_sync)

if __name__ == '__main__':
  args = get_args_parser()
  main(args)
