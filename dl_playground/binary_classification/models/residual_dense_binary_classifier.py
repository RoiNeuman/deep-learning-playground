from typing import List

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Add, Activation
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
        X = Dense(layer_units,
                  kernel_initializer='normal',
                  kernel_regularizer=l2(l2_regularizer_param),
                  bias_regularizer=l2(l2_regularizer_param)
                  )(X)
        if dropout != 0.:
            X = Dropout(dropout)(X)
        if batch_norm:
            X = BatchNormalization(axis=1)(X)
        if X.shape[1] == X_shortcut.shape[1]:
            X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

    # Output layer
    X = Dense(1, kernel_initializer='normal', activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=X, name='residual_dense_binary_classifier')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
