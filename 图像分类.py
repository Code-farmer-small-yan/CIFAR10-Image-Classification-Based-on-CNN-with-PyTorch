import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# 每批次样本数
BATCH_SIZE = 8

# todo 1.准备数据集
def create_dataset():
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset

# todo 2.搭建(卷积)神经网络
class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2, 0)
        # 全连接
        self.linear1 = nn.Linear(576, 120)
        self.linear2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # 全连接
        x = x.reshape(x.size(0), -1)
        # print(f'x.shape:{x.shape}')
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        # 输出
        return self.output(x)


# todo 3.模型训练
def train(train_dataset):
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ImageModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练
    epochs = 100 # 轮数
    for epoch_idx in range(epochs):
        # 本轮总损失, 本轮总样本数据量, 预测正确样本个数, 训练(开始)时间
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()
        for x, y in dataloader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
        # 一轮训练完毕
        print(f'轮数:{epoch_idx + 1},平均损失:{total_loss / total_samples},正确率:{total_correct / total_samples},本轮训练时间:{time.time() - start}s')
    torch.save(model.state_dict(), './model/image_model.pth')

# todo 4.模型测试
def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = ImageModel()
    model.load_state_dict(torch.load('./model/image_model.pth'))
    # 预测正确的样本个数, 总样本个数
    total_correct, total_samples = 0, 0
    for x, y in dataloader:
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=-1)
        total_correct += (y_pred == y).sum()
        total_samples += len(y)
    print(f'正确率:{total_correct / total_samples}')

# todo 5.测试
if __name__ == '__main__':
    model = ImageModel()
    # summary(model, (3, 32, 32), batch_size=BATCH_SIZE)
    # 数据集
    train_dataset, test_dataset = create_dataset()
    # 训练
    train(train_dataset)
    # 测试
    # evaluate(test_dataset)
