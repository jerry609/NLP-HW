# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def train_model(model, dataset, epochs: int, batch_size: int, learning_rate: float, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    model.train()  # 切换到训练模式
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            # **将数据移动到指定设备**
            context = context.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    return model, losses
