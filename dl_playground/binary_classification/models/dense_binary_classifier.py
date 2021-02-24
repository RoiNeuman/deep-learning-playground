from typing import List

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.python.keras.regularizers import l2


def dense_binary_classifier(input_dim: int,
                            hidden_layers_units: List[int],
                            batch_norm: bool = False,
                            dropout: float = 0.,
                            l2_regularizer_param: int = 0.01):
    model = Sequential()

    if len(hidden_layers_units):
        model.add(Dense(hidden_layers_units[0],
                        input_dim=input_dim,
                        kernel_initializer='normal',
                        kernel_regularizer=l2(l2_regularizer_param),
                        bias_regularizer=l2(l2_regularizer_param),
                        activation='relu'))
        if dropout != 0.:
            model.add(Dropout(dropout))
        if batch_norm:
            model.add(BatchNormalization(axis=1))

    for layer_units in hidden_layers_units[1:]:
        model.add(Dense(layer_units,
                        kernel_initializer='normal',
                        kernel_regularizer=l2(l2_regularizer_param),
                        bias_regularizer=l2(l2_regularizer_param),
                        activation='relu'))
        if dropout < 1:
            model.add(Dropout(dropout))
        if batch_norm:
            model.add(BatchNormalization(axis=1))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
