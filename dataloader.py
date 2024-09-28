import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import jieba
import re
from opencc import OpenCC

class textDataset(Dataset):
    print('-----data load start-----')
    '''
    自定义数据集类;加载和处理文本数据
    '''
    def __init__(self, data_file, vectorizer=None, train=True) -> None:
        self.data_file = data_file
        
        self.label_encoder = LabelEncoder()
        self.data, self.label = self.load_data(self.data_file)

        self.data = self.preprocess(self.data)

        if vectorizer is None:
            self.vectorizer = TfidfVectorizer()
        else:
            self.vectorizer = vectorizer
        
        if train:
            self.data = self.vectorizer.fit_transform(self.data)#.toarray()
            # 打印生成的特征维度
            print(f"train vectorizer特征维度：{len(self.vectorizer.get_feature_names_out())}")
        else:
            self.data = self.vectorizer.transform(self.data)
            # 打印生成的特征维度
            print(f"test vectorizer特征维度：{len(self.vectorizer.get_feature_names_out())}")

    def load_data(self, data_file):
        # test_data: pd.DataFrame = pd.read_csv(data_file,encoding='utf-8', sep="\t", header=None)
        # test_data.columns = ["新闻类别", "新闻正文"]
        train_data: pd.DataFrame = pd.read_csv(data_file,encoding='utf-8', sep="\t", header=None)
        train_data.columns = ["新闻类别", "新闻正文"]
        df_x = train_data['新闻正文']
        df_y = train_data['新闻类别']
        train_x = list(df_x) #返回字符串列表,这里是未预处理的
        train_y = self.label_encoder.fit_transform(list(df_y)) #0-9表示类别
        return train_x, train_y
    
    def preprocess(self, data):
        print('-----process data start-----')
        # 预处理
        def remove_punctuation(text):
            text = re.sub(r'[^\w\s]','',text) #.replace(' ', '')
            cc = OpenCC('t2s') #繁体字
            text =  cc.convert(text.lower()) #英文转小写
            text_l = jieba.lcut(text)
            text_l = [item for item in text_l if item!=' '] #分词

            with open('./stopwords_cn.txt','r',encoding='utf-8') as file:
                stopwords = [line.strip() for line in file.readlines()]
            text_l = [word for word in text_l if word not in stopwords] #去除停用词

            text_l2 = ' '.join(text_l) #合到一起形成字符串
            return text_l2

        clear_data = list(map(remove_punctuation, data))
        return clear_data #一个列表，里面都是字符串
    
    def __len__(self): #返回数据集大小
        return self.data.shape[0] #稀疏矩阵没有len()
    
    def __getitem__(self, idx):
        sample = self.data[idx].toarray() #将稀疏矩阵行转换为密集数组
        label = self.label[idx]

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

    def decode_label(self, encoded_labels): #方便预测时把标签0/1转换为对应的汉字标签
        return self.label_encoder.inverse_transform(encoded_labels)
    '''
    使用：
    # 创建 TextDataset 实例
    dataset = TextDataset(data_file, labels_file)

    # 假设模型预测出的结果是 [0, 1, 2]（这些是编码后的整数标签）
    predicted_labels = [0, 1, 2]

    # 使用 decode_labels 方法将这些编码的标签解码回原始的文字标签
    decoded_labels = dataset.decode_labels(predicted_labels)
    '''

# def create_dataloader(data_file,batch_size=32,shuffle=True):
#     dataset = textDataset(data_file)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#     return dataloader 

def create_dataloader(train_dataset, train_file, test_file=None, batch_size=128, shuffle=True):
    # 创建训练集的 Dataset，并在训练集上 fit TfidfVectorizer
    # train_dataset = textDataset(train_file, train=True)
    
    # 获取训练集中使用的 vectorizer
    vectorizer = train_dataset.vectorizer
    
    # 创建训练集 DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    
    if test_file:
        # 如果有测试数据，使用训练好的 vectorizer 进行 transform
        test_dataset = textDataset(test_file, vectorizer=vectorizer, train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader, vectorizer
    
    return train_dataloader, vectorizer