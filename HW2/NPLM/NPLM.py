# NPLM.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class NPLM(nn.Module):
    """
    神经概率语言模型（Neural Probabilistic Language Model）
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, context_size: int):
        super(NPLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=0.5)  # 添加 Dropout 层，防止过拟合
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # 添加批归一化层
        self.activation = nn.ReLU()  # 使用 ReLU 激活函数
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        # 对线性层进行 Xavier 初始化
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, inputs):
        """
        前向传播

        Args:
            inputs: Tensor，形状为 (batch_size, context_size)

        Returns:
            out: Tensor，形状为 (batch_size, vocab_size)
        """
        # 获取嵌入向量并调整形状
        embeds = self.embeddings(inputs)  # (batch_size, context_size, embedding_dim)
        embeds = embeds.view(inputs.size(0), -1)  # 展平成 (batch_size, context_size * embedding_dim)
        embeds = self.dropout(embeds)  # 应用 Dropout

        # 前向传播
        hidden = self.linear1(embeds)  # (batch_size, hidden_dim)
        hidden = self.batch_norm(hidden)  # 批归一化
        hidden = self.activation(hidden)  # 激活函数
        out = self.linear2(hidden)  # (batch_size, vocab_size)
        return out
