# train.py
import os
import logging
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from seqeval.metrics import classification_report
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
from collections import Counter

from HW3.config import Config
from HW3.data_loader import NERDataset, create_data_loader, load_data
from HW3.model.net import BiLSTMCRF


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        return False


def analyze_batch(batch, config, logger):
    """分析批次数据的统计信息"""
    texts, labels, lengths = batch

    # 基本形状信息
    logger.info(f"Batch shapes - Texts: {texts.shape}, Labels: {labels.shape}")
    logger.info(f"Sequence lengths - Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths) / len(lengths):.2f}")

    # 标签分布
    label_counts = Counter()
    for label_seq in labels:
        label_counts.update(label_seq.tolist())

    logger.info("Label distribution:")
    for label_id, count in label_counts.most_common():
        label_name = config.idx2tag.get(label_id, f"Unknown-{label_id}")
        logger.info(f"  {label_name}: {count}")

    return label_counts


def compute_class_weights(label_counts):
    """计算类别权重"""
    total = sum(label_counts.values())
    weights = {label: total / (len(label_counts) * count)
               for label, count in label_counts.items()}
    return weights


def evaluate(model, data_loader, config, logger):
    """评估函数"""
    model.eval()
    true_tags = []
    pred_tags = []
    total_loss = 0

    with torch.no_grad():
        for batch_idx, (texts, labels, lengths) in enumerate(data_loader):
            texts = texts.to(config.device)
            labels = labels.to(config.device)

            # 创建掩码
            mask = torch.zeros(texts.shape, dtype=torch.bool, device=config.device)
            for idx, length in enumerate(lengths):
                mask[idx, :length] = True

            # 前向传播
            emissions = model(texts, mask)
            loss = model.calc_loss(emissions, labels, mask)
            total_loss += loss.item()

            # 解码
            pred = model.decode(emissions, mask)

            # 转换为标签
            for i, length in enumerate(lengths):
                true_tags.append([config.idx2tag[t.item()] for t in labels[i][:length]])
                pred_tags.append([config.idx2tag[p] for p in pred[i][:length]])

    # 计算评估指标
    metrics = classification_report(true_tags, pred_tags, output_dict=True)
    avg_loss = total_loss / len(data_loader)

    # 打印详细评估结果
    logger.info("\nDetailed Evaluation Results:")
    logger.info(f"Average Loss: {avg_loss:.4f}")

    # 打印每个标签的详细指标
    for label, scores in sorted(metrics.items()):
        if isinstance(scores, dict):
            logger.info(f"\nLabel: {label}")
            logger.info(f"  Precision: {scores['precision']:.4f}")
            logger.info(f"  Recall: {scores['recall']:.4f}")
            logger.info(f"  F1-score: {scores['f1-score']:.4f}")
            logger.info(f"  Support: {scores['support']}")

    # 添加loss到指标字典
    metrics['loss'] = avg_loss

    return metrics


def train(model, train_loader, valid_loader, config, logger):
    """训练函数"""
    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay,
                            betas=(0.9, 0.999))

    # 设置学习率调度器
    total_steps = len(train_loader) * config.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        pct_start=0.1,  # 前10%进行warmup
        steps_per_epoch=len(train_loader),
        epochs=config.epochs,
        anneal_strategy='cos'
    )

    # 初始化混合精度训练
    scaler = torch.amp.GradScaler(
        device='cuda',
        enabled=config.use_amp
    )

    # 初始化早停
    early_stopping = EarlyStopping(patience=config.early_stopping_patience)
    best_f1 = 0

    # 分析第一个batch的数据
    first_batch = next(iter(train_loader))
    label_counts = analyze_batch(first_batch, config, logger)
    class_weights = compute_class_weights(label_counts)
    logger.info(f"Class weights: {class_weights}")

    # 开始训练循环
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        # 创建进度条
        progress_bar = tqdm(total=len(train_loader),
                            desc=f'Epoch {epoch + 1}/{config.epochs}',
                            ncols=100)

        # 批次训练
        for batch_idx, (texts, labels, lengths) in enumerate(train_loader):
            texts = texts.to(config.device)
            labels = labels.to(config.device)

            # 创建掩码
            mask = torch.zeros(texts.shape, dtype=torch.bool, device=config.device)
            for idx, length in enumerate(lengths):
                mask[idx, :length] = True

            # 混合精度训练
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu',
                                    enabled=config.use_amp):
                # 前向传播
                emissions = model(texts, mask)
                loss = model.calc_loss(emissions, labels, mask)

            # 反向传播
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # 优化器步进
            scaler.step(optimizer)
            scaler.update()

            # 更新学习率
            scheduler.step()

            # 更新统计信息
            total_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]

            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # 定期清理GPU缓存
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

                # 检查梯度
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            logger.error(f"NaN gradient detected in {name}")
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 10:
                            logger.warning(f"Large gradient norm in {name}: {grad_norm}")

        progress_bar.close()

        # 验证
        logger.info("\nRunning validation...")
        valid_metrics = evaluate(model, valid_loader, config, logger)
        valid_f1 = valid_metrics.get('macro avg', {}).get('f1-score', 0)
        valid_loss = valid_metrics.get('loss', float('inf'))

        # 记录训练信息
        logger.info(f'\nEpoch {epoch + 1}/{config.epochs}:')
        logger.info(f'Training Loss: {total_loss / len(train_loader):.4f}')
        logger.info(f'Validation Loss: {valid_loss:.4f}')
        logger.info(f'Validation F1: {valid_f1:.4f}')

        # 保存最佳模型
        if valid_f1 > best_f1:
            best_f1 = valid_f1
            save_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'config': config.__dict__
            }, save_path)
            logger.info(f'Saved best model with F1: {best_f1:.4f}')

        # 早停检查
        if early_stopping(valid_f1):
            logger.info("Early stopping triggered")
            break

        # 定期保存检查点
        if (epoch + 1) % config.save_every == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'f1': valid_f1,
                'config': config.__dict__
            }, checkpoint_path)

    return best_f1


def main():
    # 设置日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('train.log')
        ]
    )
    logger = logging.getLogger(__name__)

    # 加载配置
    config = Config()
    logger.info("Configuration loaded")
    logger.info(f"Using device: {config.device}")

    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    logger.info(f"Random seed set to {config.seed}")

    # 创建保存目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {config.checkpoint_dir}")

    # 加载数据
    logger.info("Loading data...")
    train_texts, train_labels = load_data(config.train_file, config.train_label)
    valid_texts, valid_labels = load_data(config.valid_file, config.valid_label)

    # 创建数据集和加载器
    train_dataset = NERDataset(train_texts, train_labels, config)
    valid_dataset = NERDataset(valid_texts, valid_labels, config)

    train_loader = create_data_loader(train_dataset, config.batch_size)
    valid_loader = create_data_loader(valid_dataset, config.batch_size, shuffle=False)

    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(valid_dataset)}")

    # 更新vocab_size
    config.vocab_size = train_dataset.vocab_size
    logger.info(f"Vocabulary size: {config.vocab_size}")

    # 创建模型
    model = BiLSTMCRF(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_tags=config.num_tags,
        dropout=config.dropout
    ).to(config.device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel Parameters:")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 训练模型
    logger.info("Starting training...")
    try:
        best_f1 = train(model, train_loader, valid_loader, config, logger)
        logger.info(f"Training completed! Best F1: {best_f1:.4f}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()