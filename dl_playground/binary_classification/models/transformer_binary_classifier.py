from typing import List

from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Model

from dl_playground.layers.token_position_embedding import TokenAndPositionEmbedding
from dl_playground.layers.transformer import TransformerBlock


def transformer_binary_classifier(input_dim: int,
                                  attention_heads: List[int],
                                  embed_dim: int,
                                  ff_dim: int):
    X_input = Input(shape=(input_dim, embed_dim))

    X = TokenAndPositionEmbedding(input_max_len=input_dim, embed_dim=embed_dim)(X_input)

    for num_heads in attention_heads:
        X_shortcut = X
        X = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(X)
        X = X + X_shortcut

    X = Flatten()(X)

    # Output layer
    outputs = Dense(1, kernel_initializer='normal', activation='sigmoid')(X)

    model = Model(inputs=X_input, outputs=outputs, name='transformer_binary_classifier')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
