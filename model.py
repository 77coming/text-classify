import torch
import torch.nn as nn

# class config():
class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out=self.fc2(out)
        # out=self.sigmoid(out)
        return out

class CNN(nn.Module):
    def __init__(self, input_dim, num_classes) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)

        # 计算卷积和池化后的特征数量：简单处理 (input_dim - kernel_size + 1) // pool_size
        conv_out_dim = (input_dim - 5 + 1) // 2
        conv_out_dim = (conv_out_dim - 5 + 1) // 2
        # 全连接层，输入是卷积后展平的特征数
        self.fc1 = nn.Linear(256 * conv_out_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0),-1)
         # 全连接层 + ReLU
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        # 输出层
        out = self.fc2(x)
        return out