import torch
import torch.nn as nn
import torch.nn.functional as F

"""Модифікована Модель"""

# 1. Розширений Encoder з CNN для вилучення ознак та адаптованими позиційними ембеддингами
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Створення матриці позиційних ембеддингів
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Четні індекси
        pe[:, 1::2] = torch.cos(position * div_term)  # Нечетні індекси
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_length, embedding_dim)
        seq_length = x.size(1)
        x = x + self.pe[:seq_length, :].unsqueeze(0)
        return x

class Encoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, embedding_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        # Адаптований позиційний ембеддинг
        self.positional_embedding = PositionalEmbedding(embedding_dim)

    def forward(self, x):
        x = self.conv_layers(x)  # Розмір: (batch_size, embedding_dim, seq_length)
        x = x.transpose(1, 2)    # Розмір: (batch_size, seq_length, embedding_dim)
        x = self.positional_embedding(x)
        return x

# 2. Dual-path processing block with LSTM-attention
class LSTMAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4):
        super(LSTMAttentionBlock, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(embedding_dim * 2, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
        )
        self.norm1 = nn.LayerNorm(embedding_dim * 2)
        self.norm2 = nn.LayerNorm(embedding_dim * 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = x + self.norm1(lstm_out)
        attn_out, _ = self.attention(x, x, x)
        x = x + self.norm2(self.ffn(attn_out))
        return x

# 3. Transformer decoder-based attractor calculation
class TransformerDecoderAttractor(nn.Module):
    def __init__(self, embedding_dim=256, num_queries=6, num_layers=2, num_heads=4):
        super(TransformerDecoderAttractor, self).__init__()
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, embedding_dim))
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(embedding_dim, num_heads) for _ in range(num_layers)
        ])
        self.linear = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, context):
        queries = self.query_embeddings.unsqueeze(0).expand(context.size(0), -1, -1)
        for layer in self.decoder_layers:
            queries = layer(queries, context)
        return self.linear(queries)

# 4. Triple-path processing block
class TriplePathBlock(nn.Module):
    def __init__(self, embedding_dim=256, num_heads=4):
        super(TriplePathBlock, self).__init__()
        self.intra_chunk = LSTMAttentionBlock(embedding_dim, num_heads)
        self.inter_chunk = LSTMAttentionBlock(embedding_dim, num_heads)
        self.inter_speaker = nn.TransformerEncoderLayer(embedding_dim, num_heads)

    def forward(self, x):
        x = self.intra_chunk(x)
        x = self.inter_chunk(x)
        x = self.inter_speaker(x)
        return x

# 5. Decoder
class Decoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(Decoder, self).__init__()
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # Розмір: (batch_size, embedding_dim, seq_length)
        x = self.deconv_layers(x)
        return x

# SepTDA Model
class SepTDA(nn.Module):
    def __init__(self, embedding_dim=256, separator_dim=256, num_heads=4, num_layers=8):
        super(SepTDA, self).__init__()
        self.encoder = Encoder(embedding_dim)
        self.dual_path = LSTMAttentionBlock(separator_dim, num_heads)
        self.tda = TransformerDecoderAttractor(separator_dim, num_queries=5, num_layers=2, num_heads=num_heads)
        self.triple_path = nn.ModuleList([TriplePathBlock(separator_dim, num_heads) for _ in range(num_layers)])
        self.decoder = Decoder(embedding_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # Додавання каналу
        encoded = self.encoder(x)  # Розмір: (batch_size, seq_length, embedding_dim)
        dual_path_output = self.dual_path(encoded)
        attractors = self.tda(dual_path_output)
        output = dual_path_output
        for block in self.triple_path:
            output = block(output)
        decoded = self.decoder(output)
        return decoded.squeeze(1)

# Приклад ініціалізації
model = SepTDA()
print(model)
