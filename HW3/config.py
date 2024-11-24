# config.py
import json

import torch
import os


class Config:
    def __init__(self):
        # 数据配置
        self.data_dir = "data"
        self.train_file = "data/train_corpus.txt"  # 训练文本文件
        self.train_label = "data/train_label.txt"  # 训练标签文件
        self.valid_file = "data/test_corpus.txt"  # 测试文本文件
        self.valid_label = "data/test_label.txt"  # 测试标签文件
        # 模型配置
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.num_layers = 2
        self.dropout = 0.5
        self.num_heads = 8
        # 添加梯度累积步数
        self.gradient_accumulation_steps = 2
        # 训练配置
        self.epochs = 100
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.use_amp = True  # 使用混合精度训练

        # 学习率调度配置
        self.warmup_ratio = 0.1
        self.min_lr_ratio = 0.01

        # 早停和保存配置
        self.early_stopping_patience = 5
        self.save_every = 5  # 每多少个epoch保存一次检查点

        # 路径配置
        self.checkpoint_dir = "checkpoints"
        self.log_dir = "logs"

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = 4  # DataLoader的工作进程数

        # 随机种子
        self.seed = 42

        # 序列配置
        self.max_seq_length = 512
        self.pad_token_id = 0

        # 模型特定配置
        self.num_tags = None  # 在数据加载后设置
        self.vocab_size = None  # 在数据加载后设置

        # 标签映射
        self.tag2idx = {}
        self.idx2tag = {}

        # 创建必要的目录
        self._create_directories()

        # 初始化标签映射
        self._init_tag_mappings()

    def _init_tag_mappings(self):
        """初始化标签映射"""
        try:
            # 从训练标签文件中读取标签类型
            unique_tags = set()
            with open(self.train_label, 'r', encoding='utf-8') as f:
                for line in f:
                    tags = line.strip().split()
                    unique_tags.update(tags)

            # 按照固定顺序排序标签（保证每次运行结果一致）
            sorted_tags = sorted(list(unique_tags))

            # 创建标签映射
            self.tag2idx = {tag: idx for idx, tag in enumerate(sorted_tags)}
            self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
            self.num_tags = len(self.tag2idx)

            print(f"Found {self.num_tags} unique tags: {sorted_tags}")
        except Exception as e:
            print(f"Error initializing tag mappings: {str(e)}")
            raise

    def _create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def save(self, path):
        """保存配置到文件"""
        config_dict = {k: v for k, v in self.__dict__.items()
                       if not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict))}

        # 特殊处理device
        config_dict['device'] = str(self.device)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """从文件加载配置"""
        config = cls()

        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # 更新配置
        for k, v in config_dict.items():
            if hasattr(config, k):
                # 特殊处理device
                if k == 'device':
                    config.device = torch.device(v)
                else:
                    setattr(config, k, v)

        return config

    def __str__(self):
        """返回配置的字符串表示"""
        config_str = "Model Configuration:\n"
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_str += f"{key}: {value}\n"
        return config_str