import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """
    CNN-based encoder with Global Max Pooling to handle variable sequence lengths.
    """
    def __init__(self, vocab_size, embedding_dim, latent_dim, pad_token_int):
        super(ConvEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_int)
        self.conv1 = nn.Conv1d(embedding_dim, 9, kernel_size=9)
        self.conv2 = nn.Conv1d(embedding_dim, 9, kernel_size=9)
        self.conv3 = nn.Conv1d(embedding_dim, 10, kernel_size=11)
        
        conv_out_size = self.conv1.out_channels + self.conv2.out_channels + self.conv3.out_channels
        
        self.fc_mu = nn.Linear(conv_out_size, latent_dim)
        self.fc_log_var = nn.Linear(conv_out_size, latent_dim)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1) # -> (batch, embed_dim, seq_len)
        h1 = F.relu(self.conv1(embedded))
        h2 = F.relu(self.conv2(embedded))
        h3 = F.relu(self.conv3(embedded))
        
        p1 = F.max_pool1d(h1, h1.size(2)).squeeze(2)
        p2 = F.max_pool1d(h2, h2.size(2)).squeeze(2)
        p3 = F.max_pool1d(h3, h3.size(2)).squeeze(2)
        
        h_pooled = torch.cat([p1, p2, p3], dim=1)
        mu = self.fc_mu(h_pooled)
        log_var = self.fc_log_var(h_pooled)
        return mu, log_var

class GruDecoder(nn.Module):
    """
    GRU-based decoder that uses the latent vector z at every time step.
    """
    def __init__(self, vocab_size, embedding_dim, latent_dim, hidden_dim, num_layers):
        super(GruDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim + latent_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, z):
        batch_size, seq_len = x.shape
        z_expanded = z.unsqueeze(1).expand(batch_size, seq_len, self.gru.input_size - self.embedding.embedding_dim)
        embedded = self.embedding(x)
        combined_input = torch.cat([embedded, z_expanded], dim=2)
        output, _ = self.gru(combined_input)
        prediction = self.fc_out(output)
        return prediction

class VAE(nn.Module):
    """
    The full Variational Autoencoder model.
    """
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(x, z) # Teacher forcing
        return recon_x, mu, log_var
