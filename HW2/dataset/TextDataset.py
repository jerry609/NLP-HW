# dataset.py
import os
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer
from typing import Optional


class TextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: BertTokenizer, context_size: int = 4, max_length: int = 128):
        """
        初始化数据集

        Args:
            file_path: 文本数据文件路径
            tokenizer: 用于文本编码的分词器
            context_size: 上下文窗口大小
            max_length: 文本序列的最大长度
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        self.context_size = context_size
        self.max_length = max_length
        self.tokenizer = tokenizer

        # 读取文本数据
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 分词并转换为 token ID
        tokens = self.tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True)

        # 创建上下文-目标对
        self.data = []
        if len(tokens) > context_size:
            for i in range(len(tokens) - context_size):
                context = tokens[i:i + context_size]
                target = tokens[i + context_size]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)


def get_text_data_loader(file_path: str, batch_size: int = 32, num_workers: int = 4, max_length: int = 128,
                         tokenizer: Optional[BertTokenizer] = None, language: str = 'en') -> DataLoader:
    """
    创建文本数据的 DataLoader

    Args:
        file_path: 文本数据文件路径
        batch_size: 每个批次的样本数
        num_workers: DataLoader 的工作线程数
        max_length: 文本序列的最大长度
        tokenizer: 用于文本编码的分词器（可选）
        language: 语言代码，用于动态选择分词器（默认为英文 'en'）

    Returns:
        DataLoader: 文本数据的 DataLoader 实例
    """
    # 初始化 TextDataset，并传递 tokenizer 和 language 参数
    dataset = TextDataset(file_path=file_path, max_length=max_length, tokenizer=tokenizer, language=language)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return data_loader
