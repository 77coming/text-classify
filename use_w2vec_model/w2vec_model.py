from gensim.models import Word2Vec

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
import jieba
import re
from opencc import OpenCC
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print('-----data process-----')
train_data: DataFrame = pd.read_csv("train.tsv",encoding='utf-8', sep="\t", header=None)
train_data.columns = ["新闻类别", "新闻正文"]
test_data: DataFrame = pd.read_csv("test.tsv",encoding='utf-8', sep="\t", header=None)
test_data.columns = ["新闻类别", "新闻正文"]
train_x = train_data['新闻正文']
df_train_y = list(train_data['新闻类别'])
label_encoder = LabelEncoder()
train_y = label_encoder.fit_transform(df_train_y)
test_x = test_data['新闻正文']
df_test_y =list(test_data['新闻类别'])
test_y = label_encoder.fit_transform(df_test_y)

def remove_punctuation(text):
    text = re.sub(r'[^\w\s]','',text) #.replace(' ', '')
    cc = OpenCC('t2s') #繁体字
    text =  cc.convert(text.lower()) #英文转小写
    text_l = jieba.lcut(text)
    text_l = [item for item in text_l if item!=' '] #分词

    with open('./stopwords_cn.txt','r',encoding='utf-8') as file:
        stopwords = [line.strip() for line in file.readlines()]
    text_l = [word for word in text_l if word not in stopwords] #去除停用词

    # print('1',text_l)
    # text_l2 = ' '.join(text_l) #合到一起形成字符串
    # print('2',text_l2)
    return text_l

clear_train_x = list(map(remove_punctuation, train_x))
clear_test_x = list(map(remove_punctuation, test_x))
# print(clear_test_x.shape)

print('-----word2vec model train-----')
sentence_train_x = clear_train_x
word2vec_model = Word2Vec(sentence_train_x, vector_size=300, window=5, min_count=1)
#min_count如果一个词在语料中出现的次数低于这个阈值，该词将被忽略，不被纳入词汇表
word2vec_model.wv.save('./w2vec_model/word_vectors.kv')
print('word2vec model save to \'./w2vec_model/word_vectors.kv\' ')

documents_train_x = [' '.join(text) for text in sentence_train_x ]
#['鲍勃 库西 奖','麦基 砍 28185 充满']

sentence_test_x = clear_test_x
documents_test_x = [' '.join(text) for text in sentence_test_x ]

print('-----text vectorize-----')
def text_to_vector(onesentence, word2vec_model):
    words = onesentence.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)
    
vector_X = np.array([text_to_vector(text , word2vec_model) for text in documents_train_x ])


# 定义一个简单的全连接神经网络
class TextClassificationModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TextClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # 输入维度等于词向量维度
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, input_dim, num_classes) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)

        # 计算卷积和池化后的特征数量：简单处理 (input_dim - kernel_size + 1) // pool_size
        conv_out_dim = (input_dim - 5 + 1) // 2
        conv_out_dim = (conv_out_dim - 5 + 1) // 2
        # conv_out_dim = 300
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

def text_to_vector(onesentence, word2vec_model):
    words = onesentence.split()
    word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)
    
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2vec_model):
        self.texts = texts
        self.labels = labels
        self.word2vec_model = word2vec_model

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        vector = text_to_vector(text, self.word2vec_model)  # 将文本转化为平均词向量
        return torch.tensor(vector, dtype=torch.float32), torch.tensor(label)
    
print('-----load data-----')
# 创建训练集和测试集
train_dataset = TextDataset(documents_train_x, list(train_y), word2vec_model)
test_dataset = TextDataset(documents_test_x, list(test_y), word2vec_model)

# 创建 DataLoader，不需要 collate_fn
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

print('-----start train-----')

# 初始化模型
# model = TextClassificationModel(input_dim=word2vec_model.vector_size, num_classes=10)
model = CNN(input_dim=word2vec_model.vector_size, num_classes=10)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
best_acc = 0
patience =  10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()  # 清除梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        # 累加损失
        running_loss += loss.item()
        # 预测结果
        _, predicted = torch.max(outputs, 1)  # 获取预测结果
        correct_predictions += (predicted == labels).sum().item()  # 累加正确预测的数量
        total_samples += labels.size(0)  # 总样本数

    # 计算平均损失和准确率
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_samples
    if best_acc < epoch_accuracy:
        count=0
        best_acc = epoch_accuracy
        torch.save(model.state_dict(), f'./use_w2vec_model/model_epoch{epoch+1}_TrainAcc{epoch_accuracy:.8f}.pth')
        print(f'model saved to model_epoch{epoch+1}_TrainAcc{epoch_accuracy}.pth')
            
    else:
        count+=1

    if count>=patience:
        print('early stop')
        break
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")


print('-----start test-----')
model.eval()
all_preds = []
all_labels = []
correct = 0
total = 0
with torch.no_grad():  # 评估模式不需要计算梯度
    for batch in test_loader:
        inputs, labels = batch
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())  # 存储预测结果
        all_labels.extend(labels.cpu().numpy())  # 存储真实标签
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"test acc: {100 * correct / total}%")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

test_preds = label_encoder.inverse_transform(all_preds)
pd.Series(test_preds).to_csv(rf'./use_w2vec_model/test_preds_class.csv')

