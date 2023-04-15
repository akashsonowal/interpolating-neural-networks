#!/usr/bin/env python
import tensorflow as tf


class MLPDistributedTrainer:
    def __init__(self, strategy, epochs, callbacks):
        self.strategy = strategy
        self.epochs = epochs
        self.callbacks = callbacks

        with self.strategy.scope():
            self.loss_object = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE
            )
            self.compute_loss = self._compute_loss
            self.optimizer = tf.keras.optimizers.Adam()
            self.train_loss_metric = tf.keras.metrics.MeanSquaredError(name="train_loss_metric")
            self.val_loss_metric = tf.keras.metrics.MeanSquaredError(name="val_loss_metric")

    def _compute_loss(self, labels, predictions, model_losses, global_batch_size):
        per_example_loss = self.loss_object(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss

    def train_step(self, dataset_inputs, global_batch_size, model):
        features, labels = dataset_inputs
        with tf.GradientTape() as tape:
            predictions = model(features, training=True)
            loss = self.compute_loss(labels, predictions, model.losses, global_batch_size)

        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.train_loss_metric.update_state(labels, predictions)
        return loss

    def val_step(self, dataset_inputs, model):
        features, labels = dataset_inputs
        predictions = model(features, training=False)
        self.val_loss_metric.update_state(labels, predictions)

    @tf.function
    def distributed_train_step(self, dataset_inputs, **kwargs):
        model = kwargs["model"]
        global_batch_size = kwargs["global_batch_size"]
        per_replica_losses = self.strategy.run(
            self.train_step, args=(dataset_inputs, global_batch_size, model)
        )
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_val_step(self, dataset_inputs, **kwargs):
        model = kwargs["model"]
        return self.strategy.run(self.val_step, args=(dataset_inputs, model))

    def fit(self, model, train_dataloader, val_dataloader, global_batch_size):
        for epoch in range(self.epochs):
            # Train Loop
            total_loss = 0.0
            num_batches = 0
            for dataset_inputs in train_dataloader:
                total_loss += self.distributed_train_step(
                    dataset_inputs, global_batch_size=global_batch_size, model=model
                )
                num_batches += 1
            train_loss = total_loss / num_batches
            # Validation Loop
            for dataset_inputs in val_dataloader:
                self.distributed_val_step(dataset_inputs, model=model)
            val_loss = self.val_loss_metric.result()
            self.val_loss_metric.reset_states()

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.on_train_end(
                    logs={"train_loss": train_loss.numpy(), "val_loss": val_loss.numpy()}
                )
