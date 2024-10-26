# main.py

import torch
from transformers import BertTokenizer
from HW2.dataset import TextDataset
from NPLM import NPLM
from train import train_model
from utils import save_model
import os
import matplotlib.pyplot as plt
import numpy as np

# 设置 HTTP 和 HTTPS 代理（如果需要）
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# 定义参数
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
CONTEXT_SIZE = 4
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.001
MODEL_PATH_EN = r"D:\nlp\hw1\HW2\model\nplm_en.pth"
MODEL_PATH_ZH = r"D:\nlp\hw1\HW2\model\nplm_zh.pth"
DATA_PATH_EN = r"D:\nlp\hw1\HW2\data\en.txt"  # 请将英文数据文件路径替换为您的实际路径
DATA_PATH_ZH = r"D:\nlp\hw1\HW2\data\zh.txt"  # 请将中文数据文件路径替换为您的实际路径

# 定义保存词向量的函数
def save_embeddings(word_embeddings, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for word, vector in word_embeddings.items():
            vector_str = ' '.join(map(str, vector))
            f.write(f"{word} {vector_str}\n")
    print(f"词向量已保存到 {file_path}")

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------------------
# 加载英文数据集并训练
# -------------------------------------------

# 初始化英文 tokenizer
tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建英文数据集
en_dataset = TextDataset(file_path=DATA_PATH_EN, tokenizer=tokenizer_en, context_size=CONTEXT_SIZE)

# 初始化英文模型
vocab_size_en = tokenizer_en.vocab_size
model_en = NPLM(vocab_size=vocab_size_en, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, context_size=CONTEXT_SIZE)
model_en.to(device)

# 训练英文模型
trained_model_en, en_losses = train_model(model_en, en_dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, device)
save_model(trained_model_en, MODEL_PATH_EN)

# 绘制英文模型的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), en_losses, label='English Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('English Model Training Loss')
plt.legend()
plt.savefig('english_loss.png')
plt.show()

# 提取英文模型的嵌入层权重
embedding_weights_en = trained_model_en.embeddings.weight.data.cpu().numpy()

# 获取英文 tokenizer 的词汇表并反转
vocab_en = tokenizer_en.get_vocab()
id_to_word_en = {idx: word for word, idx in vocab_en.items()}

# 组合词和对应的向量
word_embeddings_en = {}
for idx, word in id_to_word_en.items():
    vector = embedding_weights_en[idx]
    word_embeddings_en[word] = vector

# 保存英文词向量
save_embeddings(word_embeddings_en, 'english_embeddings.txt')

# -------------------------------------------
# 加载中文数据集并训练
# -------------------------------------------

# 初始化中文 tokenizer
tokenizer_zh = BertTokenizer.from_pretrained('bert-base-chinese')

# 创建中文数据集
zh_dataset = TextDataset(file_path=DATA_PATH_ZH, tokenizer=tokenizer_zh, context_size=CONTEXT_SIZE)

# 初始化中文模型
vocab_size_zh = tokenizer_zh.vocab_size
model_zh = NPLM(vocab_size=vocab_size_zh, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, context_size=CONTEXT_SIZE)
model_zh.to(device)

# 训练中文模型
trained_model_zh, zh_losses = train_model(model_zh, zh_dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, device)
save_model(trained_model_zh, MODEL_PATH_ZH)

# 绘制中文模型的损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), zh_losses, label='Chinese Training Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Chinese Model Training Loss')
plt.legend()
plt.savefig('chinese_loss.png')
plt.show()

# 提取中文模型的嵌入层权重
embedding_weights_zh = trained_model_zh.embeddings.weight.data.cpu().numpy()

# 获取中文 tokenizer 的词汇表并反转
vocab_zh = tokenizer_zh.get_vocab()
id_to_word_zh = {idx: word for word, idx in vocab_zh.items()}

# 组合词和对应的向量
word_embeddings_zh = {}
for idx, word in id_to_word_zh.items():
    vector = embedding_weights_zh[idx]
    word_embeddings_zh[word] = vector

# 保存中文词向量
save_embeddings(word_embeddings_zh, 'chinese_embeddings.txt')

# -------------------------------------------
# 绘制两个模型的损失曲线在同一张图上
# -------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(range(1, EPOCHS + 1), en_losses, label='English Training Loss')
plt.plot(range(1, EPOCHS + 1), zh_losses, label='Chinese Training Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.savefig('loss_comparison.png')
plt.show()
