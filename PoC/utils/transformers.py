import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.saving import register_keras_serializable # type: ignore
import torch
import torch.nn as nn


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
    
class Time2Vector_torch(nn.Module):
    def __init__(self, seq_len):
        super(Time2Vector_torch, self).__init__()
        self.seq_len = seq_len
        self.weights_linear = nn.Parameter(torch.Tensor(seq_len))
        self.bias_linear = nn.Parameter(torch.Tensor(seq_len))
        self.weights_periodic = nn.Parameter(torch.Tensor(seq_len))
        self.bias_periodic = nn.Parameter(torch.Tensor(seq_len))
        nn.init.uniform_(self.weights_linear)
        nn.init.uniform_(self.bias_linear)
        nn.init.uniform_(self.weights_periodic)
        nn.init.uniform_(self.bias_periodic)

    def forward(self, x):
        batch_size = x.shape[0]
        time_linear = self.weights_linear * torch.arange(self.seq_len, dtype=torch.float32).to(x.device) + self.bias_linear
        time_periodic = torch.sin(self.weights_periodic * torch.arange(self.seq_len, dtype=torch.float32).to(x.device) + self.bias_periodic)
        time_linear = time_linear.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        time_periodic = time_periodic.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        return torch.cat([time_linear, time_periodic], dim=-1)

class TransformerBlock_torch(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock_torch, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class TransformerModel(nn.Module):
    def __init__(self, seq_length, input_dim, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout):
        super(TransformerModel, self).__init__()
        self.time2vec = Time2Vector_torch(seq_length)
        self.embedding = nn.Linear(input_dim + 2, head_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock_torch(head_size, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(head_size, mlp_units[0]),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_units[0], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        time_embedding = self.time2vec(x)
        time_embedding = time_embedding.permute(0, 2, 1)  # (batch_size, 2, seq_len)
        time_embedding = time_embedding.expand(batch_size, 2, seq_len)  # Expandir para que coincida con las entradas

        x = torch.cat([x, time_embedding.permute(0, 2, 1)], dim=-1)  # (batch_size, seq_len, input_dim + 2)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.permute(1, 2, 0)  # (batch_size, embed_dim, seq_len)
        x = self.global_pool(x).squeeze(-1)
        x = self.mlp(x)
        return x