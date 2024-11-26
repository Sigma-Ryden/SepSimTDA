import torch
import torch.nn as nn
import torch.nn.functional as F

"""Model"""
# 1. Encoder
class Encoder(nn.Module):
    def __init__(self, kernel_size=16, stride=8, embedding_dim=256):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(1, embedding_dim, kernel_size, stride=stride)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv1d(x)
        return self.gelu(x)

# 2. Dual-path processing block with LSTM-attention
class LSTMAttentionBlock(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4):
        super(LSTMAttentionBlock, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, bidirectional=True, batch_first=True)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = x + self.norm1(lstm_out)
        attn_out, _ = self.attention(x, x, x)
        x = x + self.norm2(self.ffn(attn_out))
        return x

# 3. Transformer decoder-based attractor calculation
class TransformerDecoderAttractor(nn.Module):
    def __init__(self, embedding_dim=128, num_queries=6, num_layers=2, num_heads=4):
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
    def __init__(self, embedding_dim=128, num_heads=4):
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
    def __init__(self, kernel_size=16, stride=8, embedding_dim=256):
        super(Decoder, self).__init__()
        self.deconv1d = nn.ConvTranspose1d(embedding_dim, 1, kernel_size, stride=stride)

    def forward(self, x):
        return self.deconv1d(x)

# SepTDA Model
class SepTDA(nn.Module):
    def __init__(self, encoder_dim=256, separator_dim=128, kernel_size=16, stride=8, num_heads=4, num_layers=8):
        super(SepTDA, self).__init__()
        self.encoder = Encoder(kernel_size, stride, encoder_dim)
        self.dual_path = LSTMAttentionBlock(separator_dim, num_heads)
        self.tda = TransformerDecoderAttractor(separator_dim, num_queries=5, num_layers=2, num_heads=num_heads)
        self.triple_path = nn.ModuleList([TriplePathBlock(separator_dim, num_heads) for _ in range(num_layers)])
        self.decoder = Decoder(kernel_size, stride, encoder_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        dual_path_output = self.dual_path(encoded)
        attractors = self.tda(dual_path_output)
        output = dual_path_output
        for block in self.triple_path:
            output = block(output)
        decoded = self.decoder(output)
        return decoded

# Example initialization
model = SepTDA()
#print(model)


"""Dataset Preparation"""
from torch.utils.data import Dataset, DataLoader
import torchaudio

class SpeechSeparationDataset(Dataset):
    def __init__(self, mixtures, sources, transform=None):
        """
        Dataset для задачі розділення мовлення.
        Args:
            mixtures: список шляхів до мішаних аудіо файлів.
            sources: список списків шляхів до джерел мовлення (по одному на спікера).
            transform: опціональний трансформ для попередньої обробки.
        """
        self.mixtures = mixtures
        self.sources = sources
        self.transform = transform

    def __len__(self):
        return len(self.mixtures)

    def __getitem__(self, idx):
        # Завантаження мішаного аудіо
        mixture, _ = torchaudio.load(self.mixtures[idx])

        # Завантаження чистих джерел
        sources = []
        for source_path in self.sources[idx]:
            source, _ = torchaudio.load(source_path)
            sources.append(source)
        
        # Вирівнювання тривалості
        min_length = min(mixture.size(1), *[s.size(1) for s in sources])
        mixture = mixture[:, :min_length]
        sources = [s[:, :min_length] for s in sources]
        
        # Перетворення в тензор
        sources = torch.stack(sources, dim=0)  # (num_speakers, 1, time)

        # Додаткові трансформації
        if self.transform:
            mixture = self.transform(mixture)
            sources = torch.stack([self.transform(s) for s in sources])

        return mixture.squeeze(0), sources.squeeze(1)  # (time), (num_speakers, time)

# Ініціалізація датасету
mixture_paths = ["path_to_mixed_audio1.wav",
                 "path_to_mixed_audio2.wav"]

source_paths = [["path_to_source1_spk1.wav", "path_to_source1_spk2.wav"],
                ["path_to_source2_spk1.wav", "path_to_source2_spk2.wav"]]

dataset = SpeechSeparationDataset(mixture_paths, source_paths)

# Ініціалізація DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)


"""SI-SDR Loss Function"""
def si_sdr_loss(estimated, target):
    """Calculate SI-SDR loss."""
    eps = 1e-8
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True)
    projection = torch.sum(target * estimated, dim=-1, keepdim=True) / (target_energy + eps) * target
    noise = estimated - projection
    si_sdr = 10 * torch.log10((torch.sum(projection ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps))
    return -torch.mean(si_sdr)


"""Training Script"""
from torch.optim import AdamW

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SepTDA().to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=4e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (mixture, speakers) in enumerate(dataloader):
        mixture = mixture.to(device)  # Input mixture
        speakers = speakers.to(device)  # Ground truth speakers

        optimizer.zero_grad()

        # Forward pass
        estimated_sources = model(mixture.unsqueeze(1))

        # Calculate loss
        loss = sum(si_sdr_loss(estimated_sources[:, i, :], speakers[:, i, :]) for i in range(speakers.size(1)))
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # Logging
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    # Adjust learning rate
    scheduler.step(total_loss)

    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataloader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "sepsimtda_model.pth")