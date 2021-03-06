from typing import List

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model

from dl_playground.layers.residual_dense import ResidualDenseBlock


def residual_dense_binary_classifier(input_dim: int,
                                     hidden_layers_units: List[int],
                                     batch_norm: bool = False,
                                     dropout_rate: float = 0.,
                                     l2_regularizer_param: int = 0.01):
    X_input = Input(input_dim)
    X = X_input

    for layer_units in hidden_layers_units:
        X = ResidualDenseBlock(input_dim=X.shape[1],
                               ff_dim=layer_units,
                               activation='relu',
                               batch_norm=batch_norm,
                               dropout_rate=dropout_rate,
                               l2_regularizer_param=l2_regularizer_param)(X)

    # Output layer
    outputs = Dense(1, kernel_initializer='normal', activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=outputs, name='residual_dense_binary_classifier')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
