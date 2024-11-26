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


"""Dataset Preparation """
import torch
from torch.utils.data import Dataset
import torchaudio
import os

class WSJ0MixDataset(Dataset):
    """
    Dataset for WSJ0-Mix speech separation.
    Each item consists of a mixture audio and its clean sources.
    """
    def __init__(self, mixture_dir, subset="train", num_speakers=2, sample_rate=8000, segment_length=4):
        """
        Args:
            mixture_dir (str): Path to WSJ0-2Mix dataset root directory.
            subset (str): Subset to load ('train', 'val', or 'test').
            num_speakers (int): Number of speakers in the mixture (default: 2).
            sample_rate (int): Sampling rate of the audio (default: 8000 Hz).
            segment_length (int): Length of the audio segments in seconds (default: 4).
        """
        self.mixture_dir = mixture_dir
        self.subset = subset
        self.num_speakers = num_speakers
        self.sample_rate = sample_rate
        self.segment_length = segment_length

        # Paths for mixtures and sources
        self.mixture_paths = sorted(
            [os.path.join(mixture_dir, "mix", subset, f) for f in os.listdir(os.path.join(mixture_dir, "mix", subset))]
        )
        self.source_dirs = [
            os.path.join(mixture_dir, f"s{i+1}", subset) for i in range(num_speakers)
        ]
        self.source_paths = [
            [os.path.join(source_dir, os.path.basename(f)) for f in self.mixture_paths]
            for source_dir in self.source_dirs
        ]

    def __len__(self):
        return len(self.mixture_paths)

    def __getitem__(self, idx):
        # Load mixture audio
        mixture, _ = torchaudio.load(self.mixture_paths[idx])

        # Load clean sources
        sources = [torchaudio.load(self.source_paths[i][idx])[0] for i in range(self.num_speakers)]

        # Ensure each audio is the correct segment length
        total_length = self.segment_length * self.sample_rate
        mixture = torch.nn.functional.pad(
            mixture, (0, max(0, total_length - mixture.size(-1)))
        )[:, :total_length]
        sources = [
            torch.nn.functional.pad(s, (0, max(0, total_length - s.size(-1))))
            [:, :total_length]
            for s in sources
        ]

        return mixture, torch.stack(sources)

# Example usage
mixture_dir = "/path/to/wsj0-2mix"

train_dataset = WSJ0MixDataset(mixture_dir, subset="train", num_speakers=2, sample_rate=8000, segment_length=4)
#test_dataset = WSJ0MixDataset(mixture_dir, subset="test", num_speakers=2, sample_rate=8000, segment_length=4)

dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)


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