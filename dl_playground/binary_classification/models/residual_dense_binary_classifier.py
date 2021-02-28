from typing import List

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Add, Activation, ZeroPadding1D, \
    MaxPooling1D, Reshape, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2


def residual_dense_binary_classifier(input_dim: int,
                                     hidden_layers_units: List[int],
                                     batch_norm: bool = False,
                                     dropout: float = 0.,
                                     l2_regularizer_param: int = 0.01):
    X_input = Input(input_dim)
    X = X_input

    for layer_units in hidden_layers_units:
        X_shortcut = X

        # The main path
        X = Dense(layer_units,
                  kernel_initializer='normal',
                  kernel_regularizer=l2(l2_regularizer_param),
                  bias_regularizer=l2(l2_regularizer_param)
                  )(X)
        if dropout != 0.:
            X = Dropout(dropout)(X)
        if batch_norm:
            X = BatchNormalization(axis=1)(X)

        # The shortcut path
        if X_shortcut.shape[1] < layer_units:
            # Adding dimensions
            X_shortcut = Reshape((X_shortcut.shape[1], 1), input_shape=X_shortcut.shape)(X_shortcut)
            X_shortcut = ZeroPadding1D(padding=(0, layer_units - X_shortcut.shape[1]))(X_shortcut)
            X_shortcut = Flatten()(X_shortcut)
        elif X_shortcut.shape[1] > layer_units:
            # Reducing dimensions
            pool_size = X_shortcut.shape[1] - layer_units + 1
            X_shortcut = Reshape((X_shortcut.shape[1], 1), input_shape=X_shortcut.shape)(X_shortcut)
            X_shortcut = MaxPooling1D(pool_size=pool_size,
                                      strides=1,
                                      padding='valid',
                                      data_format='channels_last')(X_shortcut)
            X_shortcut = Flatten()(X_shortcut)

        # Combining the main and shortcut paths
        X = Add()([X, X_shortcut])

        # Adding non-linear activation
        X = Activation('relu')(X)

    # Output layer
    X = Dense(1, kernel_initializer='normal', activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=X, name='residual_dense_binary_classifier')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
