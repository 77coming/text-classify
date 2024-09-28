import torch
import pandas as pd
from model import NN
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import textDataset,create_dataloader

def predict(model, data_loader): #, criterion):
    print('-----predict start-----')
    writer = SummaryWriter('./runs/expriment0')

    # total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            # loss = criterion(outputs, labels)
            # total_loss+=loss.item()

            _,preds = torch.max(outputs,1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # avg_loss = total_loss/len(data_loader)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_preds, all_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('predict')
    plt.ylabel('labels')
    plt.savefig('./confusion_matrix/predict_cm.png')
    plt.close()

    writer.add_image('Confusion Matrix', torch.from_numpy(plt.imread('./confusion_matrix/predict_cm.png')).permute(2,0,1),global_step=0)
    writer.close()
    print('-----predict end-----')

    return preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='./test.tsv')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='')
    args = parser.parse_args()
    
    # print(args.file_path)
    # print(args.model_path)
    train_file = './small_train.tsv' 
    train_dataset = textDataset(train_file, train=True)
    train_loader, data_loader, vec = create_dataloader(train_dataset, train_file, args.file_path)
    
    input_size = len(vec.get_feature_names_out())
    hidden_size = 128
    output_size = 10 #10个类别

    model = NN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    outputs = predict(model, data_loader)
    # print(train_dataset.decode_label(outputs))
    outputs_array= train_dataset.decode_label(outputs)
    pd.DataFrame(outputs_array).to_csv(f'{args.output_dir}/output.csv')
    print(f'predict output saved as {args.output_dir}/output.csv')