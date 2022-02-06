import tensorflow as tf

from src.models.critic import Critic
from src.models.data_feeder import DataFeeder
from src.models.generator import Generator

class ModelEngine:
    def __init__(self, config: dict, data_feeder: DataFeeder):
        self.feeder = data_feeder

        self.generator = Generator(config)
        self.critic = Critic(config)
        self.gan = None

        self.batch_size = config['batch_size']
        self.block_len = config['block_len']
        self.num_features = config['num_features']
        self.reconstruction_lambda = config['reconstruction_loss_lambda']

        self.define_optimizer(config['optimizer'], config['learning_rate'])
        
        self.compile_critic()
        self.compile_gan()

    
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
    def wasserstein_total_loss(self, critic_loss: tf.Tensor) -> tf.Tensor:
        def total_loss_inner(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            denominator = self.batch_size * self.block_len * self.num_features
            numerator = tf.reduce_sum(tf.abs(tf.math.subtract(y_true, y_pred)))
            reconstruction_loss = numerator / denominator
            
            return critic_loss + self.reconstruction_lambda * reconstruction_loss

        return total_loss_inner

    def compile_critic(self):
        self.critic.compile(loss=self.wasserstein_critic_loss, 
                            optimizer=self.optimizer)

    def compile_gan(self):
        # will this not completely block critic?
        self.critic.trainable = False
        
        self.gan = tf.keras.Sequential()
        self.gan.add(self.generator)
        self.gan.add(self.critic)

        self.gan.compile(loss=self.wasserstein_total_loss, 
                         optimizer=self.optimizer)
        
