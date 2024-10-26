import torch
import torch.nn as nn
import torch.optim as optim


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, center_word):
        embeds = self.embeddings(center_word)
        out = self.linear(embeds)
        return out
