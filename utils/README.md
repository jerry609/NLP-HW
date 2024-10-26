
## 使用文档

### `train.py` 模块

#### `Trainer` 类

##### **初始化**

```python
trainer = Trainer(model, device, optimizer=None, criterion=None)
```

- **model** (`torch.nn.Module`): 需要训练的 PyTorch 模型。
- **device** (`torch.device`): 计算设备（如 `torch.device('cpu')` 或 `torch.device('cuda')`）。
- **optimizer** (`torch.optim.Optimizer`, 可选): 优化器，默认为 `Adam`。
- **criterion** (callable, 可选): 损失函数，默认为 `nn.CrossEntropyLoss()`。

##### **方法**

- **`set_scheduler(scheduler)`**

  设置学习率调度器。

  - **scheduler** (`torch.optim.lr_scheduler._LRScheduler`): 学习率调度器。

- **`train(train_dataset, val_dataset=None, epochs=10, batch_size=32, shuffle=True)`**

  训练模型。

  - **train_dataset** (`torch.utils.data.Dataset`): 训练数据集。
  - **val_dataset** (`torch.utils.data.Dataset`, 可选): 验证数据集。如果提供，将计算验证指标。
  - **epochs** (`int`, 可选): 训练轮数，默认为 `10`。
  - **batch_size** (`int`, 可选): 批量大小，默认为 `32`。
  - **shuffle** (`bool`, 可选): 是否打乱数据，默认为 `True`。

  **返回值：**

  - `history` (`dict`): 包含训练和验证损失及准确率的字典。

- **`evaluate(data_loader)`**

  评估模型。

  - **data_loader** (`torch.utils.data.DataLoader`): 用于评估的数据加载器。

  **返回值：**

  - 元组 (`avg_loss`, `accuracy`): 数据集上的平均损失和准确率。

---

### `visualization.py` 模块

#### **`plot_metrics` 函数**

```python
plot_metrics(metrics_dict, title="训练指标", save_path=None)
```

- **metrics_dict** (`Dict[str, Dict[str, List[float]]]`): 包含指标数据的字典。

  示例：

  ```python
  {
      '模型1': {
          'train_loss': [...],
          'val_loss': [...],
          'train_acc': [...],
          'val_acc': [...]
      },
      '模型2': {
          'train_loss': [...],
          'val_loss': [...],
          'train_acc': [...],
          'val_acc': [...]
      }
  }
  ```

- **title** (`str`, 可选): 图表标题，默认为 `"训练指标"`。
- **save_path** (`str`, 可选): 保存图像的路径。如果为 `None`，则不保存图像。

---

## 使用示例

### 使用 `Trainer` 进行训练

```python
from train import Trainer
import torch
import torch.nn as nn

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义层
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = MyModel()

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 Trainer
trainer = Trainer(model, device)

# 可选地，定义自定义的优化器和损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
trainer = Trainer(model, device, optimizer=optimizer, criterion=criterion)

# 可选地，设置学习率调度器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
trainer.set_scheduler(scheduler)

# 准备数据集
from torch.utils.data import TensorDataset

# 生成假数据
train_data = torch.randn(100, 10)
train_targets = torch.randint(0, 2, (100,))
train_dataset = TensorDataset(train_data, train_targets)

val_data = torch.randn(20, 10)
val_targets = torch.randint(0, 2, (20,))
val_dataset = TensorDataset(val_data, val_targets)

# 训练模型
history = trainer.train(train_dataset, val_dataset, epochs=10, batch_size=16)

# 在测试数据上评估
test_data = torch.randn(20, 10)
test_targets = torch.randint(0, 2, (20,))
test_dataset = TensorDataset(test_data, test_targets)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

test_loss, test_acc = trainer.evaluate(test_loader)
print(f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}')
```

### 使用 `plot_metrics` 可视化训练指标

```python
from visualization import plot_metrics

# 假设我们有两个模型的训练历史
history_model1 = {
    'train_loss': [0.9, 0.7, 0.5],
    'val_loss': [0.85, 0.75, 0.65],
    'train_acc': [0.6, 0.7, 0.8],
    'val_acc': [0.65, 0.70, 0.75]
}

history_model2 = {
    'train_loss': [0.8, 0.6, 0.4],
    'val_loss': [0.80, 0.70, 0.60],
    'train_acc': [0.65, 0.75, 0.85],
    'val_acc': [0.70, 0.75, 0.80]
}

metrics_dict = {
    '模型1': history_model1,
    '模型2': history_model2
}

# 绘制并比较指标
plot_metrics(metrics_dict, title="模型训练比较", save_path="training_metrics.png")
```

---

## 完整流程示例

将两个模块整合到完整的训练和评估流程中：

```python
import torch
from train import Trainer
from visualization import plot_metrics
from torch.utils.data import TensorDataset

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义层
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = MyModel()

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 Trainer
trainer = Trainer(model, device)

# 准备数据集
train_data = torch.randn(100, 10)
train_targets = torch.randint(0, 2, (100,))
train_dataset = TensorDataset(train_data, train_targets)

val_data = torch.randn(20, 10)
val_targets = torch.randint(0, 2, (20,))
val_dataset = TensorDataset(val_data, val_targets)

# 训练模型
history = trainer.train(train_dataset, val_dataset, epochs=10, batch_size=16)

# 可视化训练指标
metrics_dict = {'MyModel': history}
plot_metrics(metrics_dict, title="MyModel 训练指标")
```

