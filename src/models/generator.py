import tensorflow as tf
from tensorflow.keras import layers, Model, Input


class Generator(Model):
    def __init__(self, config: dict):
        super(Generator, self).__init__()
        self.encoders = []
        self.decoders = []

        filter_nums = [64, 32, 64, 128, 256, 512]
        resize_nums = [8, 16, 32, 64, 128]
        filters_encoder = filter_nums[1:]
        filters_decoder = list(reversed(filter_nums))[1:]

        activation = config['activation_generator']
        filter_size = (config['filter_length'], 1)
        dropout_rate = config['dropout']
        n_features = config['num_features']
        
        batch_size = config['batch_size']
        block_len = config['block_len']

        initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
        strides = (2, 1)

        self.input_layer = NetworkInput(n_features, initializer, batch_size, block_len)
        for num in filters_encoder:
            self.encoders.append(EncoderBlock(num, filter_size, strides, 
                                              activation, initializer, dropout_rate))
        for num, res in zip(filters_decoder, resize_nums):
            self.decoders.append(DecoderBlock(num, filter_size, res, activation, initializer))

        self.output_layer = layers.Dense(n_features, activation='tanh',
                                         kernel_initializer=initializer)

    def call(self, f0, phonemes, singers, training = False):
    # def call(self, x, training = False):
        to_concat_list = []

        x = self.input_layer(f0, phonemes, singers, training=training)
        to_concat_list.append(x)
        for encoder in self.encoders:
            x, to_concat = encoder(x, training=training)
            to_concat_list.append(to_concat)
        for num, decoder in enumerate(self.decoders):
            x = decoder(x, to_concat_list[-(num + 2)], training=training)
        output = self.output_layer(x)

        return output
    
    def summary(self):
        # For debugging
        dummy_f0 = Input(shape=(128, 42))
        dummy_phonemes = Input(shape=(128, 1))
        dummy_singers = Input(shape=(12,))
        model = Model(inputs=[dummy_f0, dummy_phonemes, dummy_singers], 
                      outputs=self.call(dummy_f0, dummy_phonemes, dummy_singers))

        return model.summary()


class NetworkInput(layers.Layer):
    def __init__(self, n_features, initializer, batch_size, 
                 block_len, input_multiplier = 1.0):
        super(NetworkInput, self).__init__()
        self.adjust_batch = batch_size
        self.adjust_block = block_len
        self.input_layers = {
            'f0': [
                layers.Dense(n_features, kernel_initializer=initializer, name='input_f0'),
                layers.BatchNormalization(scale=False, name='batch_norm_f0')
            ],
            'phonemes': [
                layers.Dense(n_features, kernel_initializer=initializer, name='input_pho'),
                layers.BatchNormalization(scale=False, name='batch_norm_pho')
            ],
            'singers': [
                layers.Dense(n_features, kernel_initializer=initializer, name='input_singers'),
                layers.BatchNormalization(scale=False, name='batch_norm_singers')
            ],

        }
        self.concatenated_layer = layers.Dense(n_features * input_multiplier, 
                                               kernel_initializer=initializer, 
                                               name='input_concat')

    def call(self, f0, phonemes, singers, training = False, features = None):
        def _input_call(feature, key):
            feature = self.input_layers[key][0](feature)
            feature = self.input_layers[key][1](feature, training=training)

            return feature
        
        _f0 = _input_call(f0, 'f0')
        _phonemes = _input_call(phonemes, 'phonemes')
        _singers = _input_call(singers, 'singers')

        _singers = tf.tile(
            tf.reshape(_singers, [self.adjust_batch, 1, -1]),
            [1, self.adjust_block, 1]
        )

        to_concat = [_f0, _phonemes, _singers]
        if features is not None:
            to_concat.insert(0, features)

        concatenated = tf.concat(to_concat, axis=-1)  # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
        # Specify shape explicitly to debug with summary() - 192 for generator, 256 for critic
        concatenated = tf.reshape(concatenated, [self.adjust_batch, 
                                                 self.adjust_block, 1, -1])
        
        output = self.concatenated_layer(concatenated)
        return output


class EncoderBlock(layers.Layer):
    def __init__(self, num_filters, filter_size, strides, 
                 activation, initializer, dropout, constraint=None):
        super(EncoderBlock, self).__init__()
        self.conv_1 = layers.Conv2D(num_filters, filter_size, strides, 
                                    padding='same', kernel_initializer=initializer, 
                                    kernel_constraint=constraint, name='conv_1')
        self.dropout_1 = layers.Dropout(rate=dropout, name='dropout_1')
        self.activation_1 = layers.Activation(activation=activation, name='activation_1')
        self.batch_norm_1 = layers.BatchNormalization(scale=False, name='batch_norm_1')
        self.block_elems = (self.conv_1, self.dropout_1, self.activation_1, self.batch_norm_1,)

    def call(self, inputs, training = False):
        x = inputs
        to_concat = None
        for layer in self.block_elems:
            if layer.name.startswith('dropout_') or layer.name.startswith('batch_norm_'):
                x = layer(x, training=training)
            else:
                x = layer(x)
                if layer.name == 'activation_1':
                    to_concat = x

        return x, to_concat


class DecoderBlock(layers.Layer):
    def __init__(self, num_filters, filter_size, resize, activation, initializer):
        super(DecoderBlock, self).__init__()
        self.resize_1 = layers.Resizing(resize, 1, interpolation='nearest')  # pylint: disable=no-member
        self.conv_1 = layers.Conv2D(num_filters, filter_size, padding='same',
                                    kernel_initializer=initializer, name='conv_1')
        self.activation_1 = layers.Activation(activation=activation, name='activation_1')
        self.batch_norm_1 = layers.BatchNormalization(scale=False, name='batch_norm_1')

        self.block_elems = (self.resize_1, self.conv_1, self.activation_1, self.batch_norm_1,)

    def call(self, inputs, to_concat, training = False):
        x = inputs

        for layer in self.block_elems:
            if layer.name.startswith('batch_norm_'):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        x = tf.concat([to_concat, x], -1)

        return x
