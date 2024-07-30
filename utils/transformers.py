import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        final_output = self.layernorm2(out1 + ffn_output)
        return final_output

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.att.key_dim,
            'num_heads': self.att.num_heads,
            'ff_dim': self.ffn.layers[0].units,
            'rate': self.dropout1.rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable()
class Time2Vector(layers.Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__(**kwargs)
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name='weight_linear', shape=(int(self.seq_len),), initializer='uniform', trainable=True)
        self.bias_linear = self.add_weight(name='bias_linear', shape=(int(self.seq_len),), initializer='uniform', trainable=True)
        self.weights_periodic = self.add_weight(name='weight_periodic', shape=(int(self.seq_len),), initializer='uniform', trainable=True)
        self.bias_periodic = self.add_weight(name='bias_periodic', shape=(int(self.seq_len),), initializer='uniform', trainable=True)

    def call(self, x):
        x = tf.math.reduce_mean(x[:,:,:4], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis=-1)
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)
        return tf.concat([time_linear, time_periodic], axis=-1)

    def get_config(self):
        config = super(Time2Vector, self).get_config()
        config.update({'seq_len': self.seq_len})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)