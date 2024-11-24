# data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import logging
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


def load_data(text_file: str, label_file: str) -> Tuple[List[List[str]], List[List[str]]]:
    """加载数据

    Args:
        text_file: 文本文件路径，每行一个句子，字符之间用空格分隔
        label_file: 标签文件路径，每行一个句子的标签，标签之间用空格分隔
    """
    texts, labels = [], []
    current_text, current_labels = [], []

    # 读取文本文件
    with open(text_file, 'r', encoding='utf-8') as f:
        text_lines = f.readlines()

    # 读取标签文件
    with open(label_file, 'r', encoding='utf-8') as f:
        label_lines = f.readlines()

    # 确保文本和标签行数相同
    assert len(text_lines) == len(label_lines), \
        f"文本行数({len(text_lines)})和标签行数({len(label_lines)})不匹配"

    # 处理每个句子
    for text_line, label_line in zip(text_lines, label_lines):
        text = text_line.strip().split()  # 分割字符
        label = label_line.strip().split()  # 分割标签

        if text and label:  # 确保不是空行
            if len(text) == 1 and len(label) == 1:  # 单个字符和标签的情况
                current_text.append(text[0])
                current_labels.append(label[0])
            else:  # 一个完整句子的情况
                if current_text:  # 如果有累积的字符，先保存
                    texts.append(current_text)
                    labels.append(current_labels)
                    current_text, current_labels = [], []

                # 保存当前句子
                texts.append(text)
                labels.append(label)

    # 处理最后一个句子
    if current_text:
        texts.append(current_text)
        labels.append(current_labels)

    # 验证数据
    for i, (text, label) in enumerate(zip(texts, labels)):
        assert len(text) == len(label), \
            f"第{i + 1}个句子的文本长度({len(text)})和标签长度({len(label)})不匹配\n" \
            f"文本: {''.join(text)}\n标签: {' '.join(label)}"

    # 打印统计信息
    logger.info(f"\n数据统计信息:")
    logger.info(f"总句子数: {len(texts)}")
    lengths = [len(t) for t in texts]
    logger.info(f"句子长度 - 最小: {min(lengths)}, 最大: {max(lengths)}, "
                f"平均: {sum(lengths) / len(lengths):.2f}")

    # 统计标签分布
    label_counter = Counter(label for seq in labels for label in seq)
    logger.info("\n标签分布:")
    for label, count in label_counter.most_common():
        logger.info(f"{label}: {count} ({count / sum(label_counter.values()) * 100:.2f}%)")

    return texts, labels

class NERDataset(Dataset):
    def __init__(self, texts: List[List[str]], labels: List[List[str]], config):
        """初始化NER数据集

        Args:
            texts: 文本列表，每个元素是字符列表
            labels: 标签列表，每个元素是标签列表
            config: 配置对象
        """
        self.texts = texts
        self.labels = labels
        self.config = config

        # 构建并保存词表
        self.char2idx = {'<PAD>': 0, '<UNK>': 1}
        self._build_vocab()

        self.vocab_size = len(self.char2idx)
        self._log_dataset_info()

    def _build_vocab(self):
        """构建词表并进行基本的统计"""
        char_counter = Counter()
        for text in self.texts:
            char_counter.update(text)

        # 按频率排序添加到词表
        for char, count in char_counter.most_common():
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)

        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    def _log_dataset_info(self):
        """记录数据集信息"""
        logger.info(f"\n数据集信息:")
        logger.info(f"样本数量: {len(self.texts)}")
        logger.info(f"词表大小: {self.vocab_size}")
        logger.info(f"标签集合: {set(label for seq in self.labels for label in seq)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """获取单个样本

        Args:
            idx: 样本索引

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (char_ids, label_ids)
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # 转换字符到索引
        char_ids = [self.char2idx.get(char, self.char2idx['<UNK>']) for char in text]

        # 转换标签到索引
        label_ids = [self.config.tag2idx[tag] for tag in label]

        return torch.tensor(char_ids), torch.tensor(label_ids)


def create_data_loader(dataset: NERDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """创建数据加载器

    Args:
        dataset: NER数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据

    Returns:
        DataLoader: 数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """处理变长序列的打包函数

    Args:
        batch: 批次数据

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (padded_texts, padded_labels, lengths)
    """
    texts, labels = zip(*batch)

    # 获取长度
    lengths = torch.tensor([len(text) for text in texts])
    max_len = max(lengths).item()

    # Padding
    padded_texts = torch.stack([
        torch.cat([text, torch.zeros(max_len - len(text))]).long()
        for text in texts
    ])

    padded_labels = torch.stack([
        torch.cat([label, torch.zeros(max_len - len(label))]).long()
        for label in labels
    ])

    return padded_texts, padded_labels, lengths


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # 测试数据加载
    texts, labels = load_data("data/train_corpus.txt", "data/train_label.txt")
    logger.info(f"\n成功加载数据")


    # 创建简单的配置对象用于测试
    class SimpleConfig:
        def __init__(self):
            self.tag2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4}


    config = SimpleConfig()

    # 创建数据集
    dataset = NERDataset(texts, labels, config)

    # 创建数据加载器
    data_loader = create_data_loader(dataset, batch_size=32)

    # 测试一个批次
    batch = next(iter(data_loader))
    logger.info(f"\n批次信息:")
    logger.info(f"文本张量形状: {batch[0].shape}")
    logger.info(f"标签张量形状: {batch[1].shape}")
    logger.info(f"序列长度: {batch[2]}")