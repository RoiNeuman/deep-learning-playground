from tensorflow.python.keras.layers import Layer, Dense, Dropout, BatchNormalization, Activation, Reshape, \
    ZeroPadding1D, Flatten, MaxPooling1D, Add, LayerNormalization
from tensorflow.python.keras.regularizers import l2


class ResidualDenseBlock(Layer):
    def __init__(self,
                 input_dim,
                 ff_dim,
                 activation: str = 'relu',
                 batch_norm: bool = False,
                 layer_norm: bool = False,
                 dropout_rate: float = 0.1,
                 l2_regularizer_param: int = 0.01):
        super(ResidualDenseBlock, self).__init__()

        self.input_dim = input_dim
        self.ff_dim = ff_dim
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        self.l2_regularizer_param = l2_regularizer_param

        # The main path components
        self.ffn = Dense(ff_dim,
                         kernel_initializer='normal',
                         kernel_regularizer=l2(l2_regularizer_param),
                         bias_regularizer=l2(l2_regularizer_param)
                         )
        self.dropout = Dropout(dropout_rate)
        self.batch_norm_layer = BatchNormalization(axis=1)

        # The shortcut path components
        self.reshape = Reshape((input_dim, 1), input_shape=(None, input_dim))
        self.padding = ZeroPadding1D(padding=(0, ff_dim - input_dim))
        self.pooling = MaxPooling1D(pool_size=input_dim - ff_dim + 1,
                                    strides=1,
                                    padding='valid',
                                    data_format='channels_last')
        self.flatten = Flatten()

        self.add = Add()
        self.activation = Activation(activation)
        self.layernorm_layer = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, **kwargs):
        X_shortcut = inputs

        # The main path
        X = self.ffn(inputs)
        if self.dropout_rate != 0.:
            X = self.dropout(X, **kwargs)
        if self.batch_norm:
            X = self.batch_norm_layer(X)

        # The shortcut path
        if self.input_dim < self.ff_dim:
            # Adding dimensions
            X_shortcut = self.reshape(X_shortcut)
            X_shortcut = self.padding(X_shortcut)
            X_shortcut = self.flatten(X_shortcut)
        elif self.input_dim > self.ff_dim:
            # Reducing dimensions
            X_shortcut = self.reshape(X_shortcut)
            X_shortcut = self.pooling(X_shortcut)
            X_shortcut = self.flatten(X_shortcut)

        # Combining the main and shortcut paths
        X = self.add([X, X_shortcut])
        out = self.activation(X)

        if self.layer_norm:
            out = self.layernorm_layer(out)

        return out
