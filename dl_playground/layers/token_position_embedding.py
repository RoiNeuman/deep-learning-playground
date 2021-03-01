import tensorflow as tf
from tensorflow.python.keras.layers import Layer, Embedding


class TokenAndPositionEmbedding(Layer):
    """
    From: https://keras.io/examples/nlp/text_classification_with_transformer/
    """

    def __init__(self, input_max_len, embed_dim, vocab_size: int or None = None):
        super(TokenAndPositionEmbedding, self).__init__()
        self.input_max_len = input_max_len
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        if vocab_size:
            self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=input_max_len, output_dim=embed_dim)

    def call(self, inputs, **kwargs):
        positions = tf.range(start=0, limit=self.input_max_len, delta=1)
        positions = self.pos_emb(positions)
        if self.vocab_size:
            x = self.token_emb(inputs)
        else:
            x = inputs
        return x + positions
