# visualization.py
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_losses(losses_dict: Dict[str, List[float]], title: str = "Training Loss", save_path: str = None):
    """
    绘制并保存训练损失曲线。

    Args:
        losses_dict (Dict[str, List[float]]): 各模型损失数据字典，键为模型名称，值为损失列表
        title (str): 图表标题
        save_path (str): 保存图片的路径，默认为 None
    """
    plt.figure(figsize=(10, 5))
    for label, losses in losses_dict.items():
        plt.plot(losses, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()

    if save_path:
        plt.savefig(save_path)  # 保存图像
    plt.show()
