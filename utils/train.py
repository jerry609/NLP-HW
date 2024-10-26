import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, device, optimizer=None, criterion=None):
        """
        初始化 Trainer。

        参数：
            model (torch.nn.Module): 需要训练的模型。
            device (torch.device): 训练设备。
            optimizer (torch.optim.Optimizer, 可选): 优化器，默认为 Adam 优化器。
            criterion (callable, 可选): 损失函数，默认为 nn.CrossEntropyLoss()。
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer if optimizer else optim.Adam(self.model.parameters())
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()
        self.scheduler = None  # 学习率调度器

    def set_scheduler(self, scheduler):
        """
        设置学习率调度器。

        参数：
            scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器。
        """
        self.scheduler = scheduler

    def train(self, train_dataset, val_dataset=None, epochs=10, batch_size=32, shuffle=True):
        """
        训练模型。

        参数：
            train_dataset (torch.utils.data.Dataset): 训练数据集。
            val_dataset (torch.utils.data.Dataset, 可选): 验证数据集，默认为 None。
            epochs (int, 可选): 训练轮数，默认为 10。
            batch_size (int, 可选): 批量大小，默认为 32。
            shuffle (bool, 可选): 是否打乱数据，默认为 True。

        返回：
            dict: 包含训练和验证损失及准确率的历史记录。
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

        history = {'train_loss': [], 'train_acc': []}
        if val_loader:
            history['val_loss'] = []
            history['val_acc'] = []

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(accuracy)

            # 验证阶段
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

            if self.scheduler:
                self.scheduler.step()

        return history

    def evaluate(self, data_loader):
        """
        评估模型。

        参数：
            data_loader (torch.utils.data.DataLoader): 数据加载器。

        返回：
            tuple: 包含损失和准确率的元组。
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        return avg_loss, accuracy
