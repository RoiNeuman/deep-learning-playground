from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Layer, MultiHeadAttention, Dense, LayerNormalization, Dropout


class TransformerBlock(Layer):
    """
    From: https://keras.io/examples/nlp/text_classification_with_transformer/
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, **kwargs):
        attention_output = self.attention(inputs, inputs)
        attention_output1 = self.dropout1(attention_output, **kwargs)
        out1 = self.layernorm1(inputs + attention_output1)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, **kwargs)
        return self.layernorm2(out1 + ffn_output)
