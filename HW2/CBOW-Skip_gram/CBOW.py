import torch
import torch.nn as nn
import torch.optim as optim


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, context):
        # 计算上下文单词的嵌入表示，并取平均
        embeds = torch.mean(self.embeddings(context), dim=1)
        out = torch.relu(self.linear1(embeds))
        out = self.linear2(out)
        return out
