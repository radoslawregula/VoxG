from datetime import datetime
import h5py
import logging
from math import ceil
import platform
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from src.models.critic import Critic
from src.models.data_feeder import DataFeeder
from src.models.generator import Generator
from src.utils.helpers import to_wider_limits, to_narrow_limits

tf.get_logger().setLevel('WARNING')
logger = logging.getLogger(__name__)
DELTA: float = 1e-12


class ModelEngine:
    def __init__(self, config: dict, data_feeder: DataFeeder):
        self.feeder = data_feeder
        
        # Models
        self.generator = Generator(config)
        self.critic = Critic(config)

        # Hyperparameters
        self.epochs = config['epochs']
        self.validate_every = config['validate_every']
        self.critic_inloops = config['critic_inloops']  # Assert >= 1
        self.reconstruction_lambda = config['reconstruction_loss_lambda']

        # Optimizers
        self.critic_optimizer = None
        self.generator_optimizer = None
        self.define_optimizers(config['optimizer'], config['learning_rate'])

        # Saving 
        self.save_every = config['save_every']
        self.model_output_dir = config['model_output_dir']

    def _setup_training_dir(self) -> str:
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        host = platform.node()
        training_dir = f'training_{host}_{timestamp}'

        training_dirpath = os.path.join(self.model_output_dir, training_dir)
        
        try:
            os.makedirs(training_dirpath, exist_ok=False)
        except IOError as ioe:
            logger.error(f'Failed to create training directory at {training_dirpath}.')
            raise ioe
        
        return training_dirpath

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
        denominator = self.feeder.get_average_by()
        numerator = tf.reduce_sum(tf.abs(tf.math.subtract(y_true, y_pred)))
        reconstruction_loss = numerator / denominator
        critic_loss = tf.reduce_mean(critic_score + DELTA)  # Shouldn't that be a whole
                                                            # critic loss formula?
        
        return critic_loss + self.reconstruction_lambda * reconstruction_loss
    
    @staticmethod
    def _is_divisible(what: int, by: int) -> bool:
        return what % by == 0
    
    def train(self):
        this_training_dir = None  # Don't create dirs until the first save
        critic_loss_summary = []
        final_loss_summary = []
        valid_critic_loss_summary = []
        valid_final_loss_summary = []

        for epoch_idx in range(self.epochs):
            epoch = epoch_idx + 1
            logger.info(f'Epoch {epoch}:')

            critic_loss_per_inloop = tf.keras.metrics.Mean()
            critic_loss_per_epoch = tf.keras.metrics.Mean()
            final_loss_per_epoch = tf.keras.metrics.Mean()

            valid_critic_loss_per_epoch = tf.keras.metrics.Mean()
            valid_final_loss_per_epoch = tf.keras.metrics.Mean()

            train_generator = self.feeder.train_data_generator()
            valid_generator = self.feeder.valid_data_generator()

            progbar = tf.keras.utils.Progbar(target=self.feeder.per_epoch)

            for idx, (features_real, f0, phonemes, singers) in enumerate(train_generator):
                # _update_generator = self._is_divisible(idx + 1, self.critic_inloops + 1)
                # _update_critic = not _update_generator

                # From original WGANSing
                inloops = self.critic_inloops * 3 if epoch < 26 or epoch % 100 == 0 \
                          else self.critic_inloops

                # if _update_critic:
                for _ in range(inloops):
                    with tf.GradientTape() as critic_tape:
                        features_fake = self.generator(f0, phonemes, singers, training=True)
                        score_fake = self.critic(features_fake, f0, phonemes, 
                                                 singers, training=True)
                        score_real = self.critic(to_wider_limits(features_real), f0, 
                                                 phonemes, singers, training=True)
                        c_loss = self.wasserstein_critic_loss(score_real, score_fake)
                    c_grads = critic_tape.gradient(c_loss, self.critic.trainable_weights)
                    self.critic_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_weights))

                    critic_loss_per_inloop.update_state(c_loss)

                # if _update_generator:
                with tf.GradientTape() as gen_tape:
                    features_fake = self.generator(f0, phonemes, singers, training=True)
                    score = self.critic(features_fake, f0, phonemes, singers, training=True)
                    g_loss = self.wasserstein_total_loss(features_real, 
                                                            to_narrow_limits(features_fake), 
                                                            score)
                g_grads = gen_tape.gradient(g_loss, self.generator.trainable_weights)
                self.generator_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

                final_loss_per_epoch.update_state(g_loss)
                critic_loss_per_epoch.update_state(critic_loss_per_inloop.result())

                progbar.update(current=idx+1, values=[
                    ('Critic Loss', critic_loss_per_inloop.result()),
                    ('Final Loss', g_loss.numpy())
                ])
                critic_loss_per_inloop.reset_state()
            
            valid_progbar = tf.keras.utils.Progbar(target=self.feeder.per_epoch_valid)

            if self._is_divisible(epoch, self.validate_every):
                for idx, (features_real, f0, phonemes, singers) in enumerate(valid_generator):
                    features_fake = self.generator(f0, phonemes, singers, training=False)
                    score_fake = self.critic(features_fake, f0, phonemes, singers, training=False)
                    score_real = self.critic(to_wider_limits(features_real), f0, 
                                             phonemes, singers, training=False)

                    val_c_loss = self.wasserstein_critic_loss(score_real, score_fake)
                    val_g_loss = self.wasserstein_total_loss(features_real, 
                                                             to_narrow_limits(features_fake),
                                                             score_fake)
                    
                    valid_critic_loss_per_epoch.update_state(val_c_loss)
                    valid_final_loss_per_epoch.update_state(val_g_loss)

                    valid_progbar.update(current=idx+1, values=[
                            ('Critic Loss', val_c_loss.numpy()),
                            ('Final Loss', val_g_loss.numpy())
                        ])

                # Update per-epoch metrics
                valid_critic_loss_summary.append(valid_critic_loss_per_epoch.result())
                valid_final_loss_summary.append(valid_final_loss_per_epoch.result()) 

            # Update per-epoch metrics
            critic_loss_summary.append(critic_loss_per_epoch.result())
            final_loss_summary.append(final_loss_per_epoch.result())           
            
            if self._is_divisible(epoch, self.save_every):
                if this_training_dir is None:
                    this_training_dir = self._setup_training_dir()
                self.save_generator_weights(this_training_dir, epoch)

                self.draw_learning_curves(this_training_dir, epoch,
                                          critic_loss_summary,
                                          final_loss_summary,
                                          valid_critic_loss_summary,
                                          valid_final_loss_summary)
    
    def save_generator_weights(self, training_dir: str, epoch_number: int):
        filename = f'checkpoint-e{epoch_number}.h5'
        filepath = os.path.join(training_dir, filename)
        location_info = filepath.split(self.model_output_dir)[-1][1:]
        
        logger.info(f'Checkpoint: saving generator weights to {location_info}.')
        self.generator.save_weights(filepath, overwrite=True, save_format='h5')
    
    def draw_learning_curves(self, training_dir: str, epoch_number: int, *args):
        train_critic, train_final, val_critic, val_final = args
        training_x_axis = list(range(1, epoch_number + 1))
        validation_x_axis = list(filter(lambda x: self._is_divisible(x, self.validate_every), 
                                        training_x_axis))
        
        descriptors = [
            {
                'x': training_x_axis, 
                'y': train_critic, 
                'losstype': 'Critic', 
                'mode': 'training', 
                'color': 'firebrick',
                'fpath_img': os.path.join(training_dir, 'train-critic.png'),
                'fpath_csv': os.path.join(training_dir, 'train-critic-data.csv')
            },
            {
                'x': training_x_axis, 
                'y': train_final, 
                'losstype': 'Final', 
                'mode': 'training', 
                'color': 'slateblue',
                'fpath_img': os.path.join(training_dir, 'train-final.png'),
                'fpath_csv': os.path.join(training_dir, 'train-final-data.csv')
            },
            {
                'x': validation_x_axis, 
                'y': val_critic, 
                'losstype': 'Critic', 
                'mode': 'validation', 
                'color': 'firebrick',
                'fpath_img': os.path.join(training_dir, 'val-critic.png'),
                'fpath_csv': os.path.join(training_dir, 'val-critic-data.csv')
            },
            {
                'x': validation_x_axis, 
                'y': val_final, 
                'losstype': 'Final', 
                'mode': 'validation', 
                'color': 'slateblue',
                'fpath_img': os.path.join(training_dir, 'val-final.png'),
                'fpath_csv': os.path.join(training_dir, 'val-final-data.csv')
            }
        ]

        def _draw_single_figure(descriptor: dict):
            if len(descriptor['y']) < 2:
                return

            plt.figure()
            plt.plot(descriptor['x'], descriptor['y'], 
                     color=descriptor['color'])
            plt.xlim([descriptor['x'][0], descriptor['x'][-1]])
            plt.xticks(descriptor['x'][::ceil(len(descriptor['x']) / 10)])
            plt.grid(True)
            plt.xlabel('Epochs')
            plt.ylabel('Loss value')
            plt.title(f'{descriptor["losstype"]} loss for ' 
                      f'{descriptor["mode"]}: epoch {epoch_number}')
            plt.savefig(descriptor['fpath_img'])

            # Save raw data for future visualization purposes
            loss_numpy = [t.numpy() for t in descriptor['y']]
            df = pd.DataFrame({'Epoch': descriptor['x'], 'Loss': loss_numpy})
            df.to_csv(descriptor['fpath_csv'], index=False)

        for dsc in descriptors:
            _draw_single_figure(dsc)

        logger.info(f'Visualizer: saving learning curves to {os.path.basename(training_dir)}.')
