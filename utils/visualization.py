import matplotlib.pyplot as plt
from typing import List, Dict

def plot_metrics(metrics_dict: Dict[str, Dict[str, List[float]]], title: str = "训练指标", save_path: str = None):
    """
    绘制并保存训练和验证的损失和准确率曲线。

    参数：
        metrics_dict (Dict[str, Dict[str, List[float]]]): 包含指标数据的字典。
            示例：{'模型1': {'train_loss': [...], 'val_loss': [...], 'train_acc': [...], 'val_acc': [...]}, ...}
        title (str): 图表标题。
        save_path (str): 保存图片的路径，默认为 None。
    """
    epochs = range(1, len(next(iter(metrics_dict.values()))['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    for label, metrics in metrics_dict.items():
        plt.plot(epochs, metrics.get('train_loss', []), label=f'{label} 训练损失')
        if 'val_loss' in metrics:
            plt.plot(epochs, metrics['val_loss'], label=f'{label} 验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('损失曲线')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    for label, metrics in metrics_dict.items():
        plt.plot(epochs, metrics.get('train_acc', []), label=f'{label} 训练准确率')
        if 'val_acc' in metrics:
            plt.plot(epochs, metrics['val_acc'], label=f'{label} 验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.title('准确率曲线')
    plt.legend()

    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
