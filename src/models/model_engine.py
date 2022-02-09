import logging

import numpy as np
import tensorflow as tf

from src.models.critic import Critic
from src.models.data_feeder import DataFeeder
from src.models.generator import Generator
from src.utils.helpers import to_wider_limits, to_narrow_limits

logger = logging.getLogger(__name__)


class ModelEngine:
    def __init__(self, config: dict, data_feeder: DataFeeder):
        self.feeder = data_feeder

        self.generator = Generator(config)
        self.critic = Critic(config)
        self.gan = None

        self.epochs = config['epochs']
        # Assert >= 1
        self.critic_inloops = config['critic_inloops']
        self.num_features = config['num_features']
        self.reconstruction_lambda = config['reconstruction_loss_lambda']

        self.define_optimizer(config['optimizer'], config['learning_rate'])

    
    def define_optimizer(self, optimizer: str, lr: float):
        if optimizer == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        else:
            raise NotImplementedError

    def wasserstein_critic_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        DELTA: float = 1e-12
        return tf.math.subtract(tf.reduce_mean(y_true + DELTA), 
                                tf.reduce_mean(y_pred + DELTA))
    
    #  TODO: not forget -0.5*2
    def wasserstein_total_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                               critic_loss: tf.Tensor) -> tf.Tensor:
        denominator = self.batch_size * self.block_len * self.num_features
        numerator = tf.reduce_sum(tf.abs(tf.math.subtract(y_true, y_pred)))
        reconstruction_loss = numerator / denominator
        
        return critic_loss + self.reconstruction_lambda * reconstruction_loss

    def compile_critic(self):
        self.critic.compile(loss=self.wasserstein_critic_loss, 
                            optimizer=self.optimizer)
    
    def compile_generator(self):
        self.generator.compile(loss=self.wasserstein_critic_loss, 
                               optimizer=self.optimizer)

    def build_gan(self):
        # will this not completely block critic? Not if compile_critic called first (?)
        self.critic.trainable = False

        input_f0 = tf.keras.Input(shape=(128, 42))
        input_phonemes = tf.keras.Input(shape=(128, 1))
        input_singers = tf.keras.Input(shape=(12,))
        input_training = tf.keras.Input(shape=(1,), dtype=tf.bool)
        x = self.generator(input_f0, input_phonemes, input_singers, input_training)
        output = self.critic(x, input_f0, input_phonemes, input_singers, input_training)
        self.gan = tf.keras.Model(inputs=[input_f0, input_phonemes, input_singers], outputs=output)
    
    def compile_all(self):
        # self.compile_generator()
        self.compile_critic()
        self.build_gan()
    
    def train(self):
        self.compile_all()
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
                    with tf.GradientTape() as tape:
                        features_fake = self.generator(f0, phonemes, singers, training=True)
                        c_loss = self.wasserstein_critic_loss(to_wider_limits(features_real), 
                                                              features_fake)
                    grads = tape.gradient(c_loss, self.critic.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

                    critic_losses.append(c_loss)

                if _update_generator:
                    with tf.GradientTape() as tape:
                        score = self.gan(f0, phonemes, singers, training=True)
                        features_fake = self.generator(f0, phonemes, singers, training=True)
                        g_loss = self.wasserstein_total_loss(features_real, 
                                                             to_narrow_limits(features_fake), 
                                                             score)
                    grads = tape.gradient(g_loss, self.gan.trainable_weights)
                    self.optimizer.apply_gradients(zip(grads, self.gan.trainable_weights))

                    final_losses.append(g_loss)
                    critic_mean_losses.append(np.mean(critic_losses))
                    critic_losses.clear()

                    logger.info(f'Iteration {idx + 1}'
                                f' - Critic Loss: {critic_mean_losses[-1]}'
                                f' - Overall Loss: {final_losses[-1]}')
                    
                    
                





        
