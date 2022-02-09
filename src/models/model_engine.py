import logging

import numpy as np
import tensorflow as tf

from src.models.critic import Critic
from src.models.data_feeder import DataFeeder
from src.models.generator import Generator
from src.utils.helpers import to_wider_limits, to_narrow_limits

logger = logging.getLogger(__name__)
DELTA: float = 1e-12


class ModelEngine:
    def __init__(self, config: dict, data_feeder: DataFeeder):
        self.feeder = data_feeder

        self.generator = Generator(config)
        self.critic = Critic(config)

        self.epochs = config['epochs']
        # Assert >= 1
        self.critic_inloops = config['critic_inloops']
        self.num_features = config['num_features']
        self.reconstruction_lambda = config['reconstruction_loss_lambda']

        self.critic_optimizer = None
        self.generator_optimizer = None
        self.define_optimizers(config['optimizer'], config['learning_rate'])

    
    def define_optimizers(self, optimizer: str, lr: float):
        if optimizer == 'rmsprop':
            self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
            self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer == 'adam':
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'sgd':
            self.critic_optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
            self.generator_optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            raise NotImplementedError

    def wasserstein_critic_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.math.subtract(tf.reduce_mean(y_true + DELTA), 
                                tf.reduce_mean(y_pred + DELTA))
    
    def wasserstein_total_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                               critic_score: tf.Tensor) -> tf.Tensor:
        denominator = self.feeder.batch_size * self.feeder.block_len * self.num_features
        numerator = tf.reduce_sum(tf.abs(tf.math.subtract(y_true, y_pred)))
        reconstruction_loss = numerator / denominator
        critic_loss = tf.reduce_mean(critic_score + DELTA)
        
        return critic_loss + self.reconstruction_lambda * reconstruction_loss
    
    def train(self):
        critic_losses = []
        critic_mean_losses = []
        final_losses = []

        for epoch_idx in range(self.epochs):
            epoch = epoch_idx + 1
            logger.info(f'Epoch {epoch}:')
            train_generator = self.feeder.train_data_generator()
            valid_generator = self.feeder.valid_data_generator()

            for idx, (features_real, f0, phonemes, singers) in enumerate(train_generator):
                _update_generator = (idx + 1) % (self.critic_inloops + 1) == 0
                _update_critic = not _update_generator

                if _update_critic:
                    with tf.GradientTape() as critic_tape:
                        features_fake = self.generator(f0, phonemes, singers, training=True)
                        score_fake = self.critic(features_fake, f0, phonemes, 
                                                 singers, training=True)
                        score_real = self.critic(to_wider_limits(features_real), f0, 
                                                 phonemes, singers, training=True)
                        c_loss = self.wasserstein_critic_loss(score_real, score_fake)
                    c_grads = critic_tape.gradient(c_loss, self.critic.trainable_weights)
                    self.critic_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_weights))

                    critic_losses.append(c_loss)

                if _update_generator:
                    with tf.GradientTape() as gen_tape:
                        features_fake = self.generator(f0, phonemes, singers, training=True)
                        score = self.critic(features_fake, f0, phonemes, singers, training=True)
                        g_loss = self.wasserstein_total_loss(features_real, 
                                                             to_narrow_limits(features_fake), 
                                                             score)
                    g_grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)
                    self.generator_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

                    final_losses.append(g_loss)
                    critic_mean_losses.append(np.mean(critic_losses))
                    critic_losses.clear()

                    logger.info(f'Iteration {idx + 1}'
                                f' - Critic Loss: {critic_mean_losses[-1]}'
                                f' - Overall Loss: {final_losses[-1]}')
                    
                    
                





        
