# utils.py
import logging
import torch
import numpy as np
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2


def setup_logger(name, log_file, level=logging.INFO):
    """设置日志器"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def load_data(corpus_file, label_file, encoding='utf-8'):
    """加载数据并进行基本的数据检查"""
    texts, labels = [], []

    # 读取文本
    with open(corpus_file, 'r', encoding=encoding) as f:
        for line in f:
            text = list(line.strip())
            if text:  # 确保不是空行
                texts.append(text)

    # 读取标签
    with open(label_file, 'r', encoding=encoding) as f:
        for line in f:
            label = line.strip().split()
            if label:  # 确保不是空行
                labels.append(label)

    # 数据检查
    assert len(texts) == len(labels), \
        f"文本数量 ({len(texts)}) 与标签数量 ({len(labels)}) 不匹配"

    for i, (text, label) in enumerate(zip(texts, labels)):
        assert len(text) == len(label), \
            f"第 {i} 行的文本长度 ({len(text)}) 与标签长度 ({len(label)}) 不匹配"

    return texts, labels


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def compute_entity_level_metrics(true_tags, pred_tags):
    """计算实体级别的评估指标"""
    results = classification_report(true_tags, pred_tags,
                                    scheme=IOB2,
                                    mode='strict',
                                    output_dict=True)

    # 计算每个实体类型的指标
    entity_metrics = {}
    for entity_type in ['PER', 'LOC', 'ORG']:
        metrics = {}
        for prefix in ['B-', 'I-']:
            tag = prefix + entity_type
            if tag in results:
                for metric in ['precision', 'recall', 'f1-score']:
                    key = f'{prefix}{entity_type}_{metric}'
                    metrics[key] = results[tag][metric]
        entity_metrics[entity_type] = metrics

    return {
        'entity_metrics': entity_metrics,
        'micro_avg': results['micro avg'],
        'macro_avg': results['macro avg'],
        'weighted_avg': results['weighted avg']
    }


def adjust_learning_rate(optimizer, current_step, total_steps, warmup_steps, max_lr):
    """调整学习率"""
    if current_step < warmup_steps:
        # 预热阶段
        lr = max_lr * current_step / warmup_steps
    else:
        # 衰减阶段
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        lr = max_lr * (1 + np.cos(np.pi * progress)) / 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def save_model(model, optimizer, scheduler, epoch, best_metric, path):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric
    }
    torch.save(checkpoint, path)


def load_model(model, optimizer, scheduler, path):
    """加载模型检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_metric']


def analyze_errors(true_tags, pred_tags, texts):
    """分析预测错误"""
    error_cases = []
    for text, true_seq, pred_seq in zip(texts, true_tags, pred_tags):
        if true_seq != pred_seq:
            errors = []
            for i, (true_tag, pred_tag) in enumerate(zip(true_seq, pred_seq)):
                if true_tag != pred_tag:
                    errors.append({
                        'position': i,
                        'char': text[i],
                        'true_tag': true_tag,
                        'pred_tag': pred_tag
                    })
            error_cases.append({
                'text': ''.join(text),
                'errors': errors
            })
    return error_cases