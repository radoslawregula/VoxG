import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from src.models.generator import NetworkInput, EncoderBlock


class Critic(Model):
    def __init__(self, config: dict):
        super(Critic, self).__init__()
        self.layers_critic = []

        filter_nums = [64, 64, 128, 128, 256, 256, 512]
        filters_critic = filter_nums

        activation = config['activation_critic']
        filter_size = (config['filter_length'], 1)
        dropout_rate = config['dropout']
        n_features = config['num_features']
        
        batch_size = config['batch_size']
        block_len = config['block_len']

        initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
        strides = (2, 1)

        self.input_layer = NetworkInput(n_features, initializer,
                                        batch_size, block_len, 
                                        input_multiplier=2.0)
        for num in filters_critic:
            self.layers_critic.append(EncoderBlock(num, filter_size, strides, 
                                                   activation, initializer, 
                                                   dropout_rate))
        self.output_layer = layers.Dense(1, kernel_initializer=initializer)

    def call(self, features, f0, phonemes, singers, training = False):
        x = self.input_layer(f0, phonemes, singers, 
                             training=training, features=features)
        for net_layer in self.layers_critic:
            x, _ = net_layer(x, training=training)
        
        x = tf.squeeze(x)
        output = self.output_layer(x)
        output = tf.squeeze(output)
        
        return output

    def summary(self):
        # For debugging
        dummy_features = Input(shape=(128, 64))
        dummy_f0 = Input(shape=(128, 42))
        dummy_phonemes = Input(shape=(128, 1))
        dummy_singers = Input(shape=(12,))
        model = Model(inputs=[dummy_features, dummy_f0, 
                              dummy_phonemes, dummy_singers], 
                      outputs=self.call(dummy_features, dummy_f0, 
                                        dummy_phonemes, dummy_singers))

        return model.summary()
