from model import NN
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

from tqdm import tqdm #进度条
from torch.utils.tensorboard import SummaryWriter
from dataloader import textDataset,create_dataloader
from sklearn.metrics import accuracy_score

base_log_dir = "./logs"

def get_new_log_dir(base_log_dir):
    existing_versions = [d for d in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir,d))]
    existing_versions = [int(d.replace('version','')) for d in existing_versions if d.startswith('version')]

    if existing_versions:
        new_version = max(existing_versions) + 1
    else:
        new_version = 0
    
    new_log_dir = os.path.join(base_log_dir, f'version{new_version}')
    return new_log_dir

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0 
    all_preds = []
    all_labels = []
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        
        outputs = model(inputs)
        # print('outputs.shape',outputs.shape)
        #outputs.shape torch.Size([100, 1, 10])
        #去掉第2个维度
        outputs = outputs.squeeze(1)
        # print('outputs.shape',outputs.shape)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        total_loss += loss.item()

    avg_loss= total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss,acc
  
def evaluate(model, test_loader, criterion):
    print('-----evaluate start-----')
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            #去掉第2个维度
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss/len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss,acc



def main():
    log_dir = get_new_log_dir(base_log_dir)
    writer = SummaryWriter(log_dir)
    hidden_size = 128
    output_size = 10 #10个类别
    num_epochs = 50
    batch_size = 128
    lr = 0.001
    patience = 5 #早停
    train_file = './small_train.tsv'
    test_file =  './small_test.tsv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default=train_file)
    parser.add_argument('--val_file', type=str, default=test_file)
    parser.add_argument('--num_epochs', type=int, default = num_epochs)
    parser.add_argument('--batch_size', type=int, default = batch_size)
    parser.add_argument('--lr', type=float, default = lr)
    parser.add_argument('--patience', type=int, default = patience)
    args = parser.parse_args()
    # train_dataset = textDataset('./small_train.tsv')
    # test_dataset = textDataset('./small_test.tsv')
    # input_size = len(train_dataset.vectorizer.get_feature_names_out())

    # train_loader = create_dataloader('./small_train.tsv',batch_size=batch_size,shuffle=True)
    # test_loader = create_dataloader('./small_test.tsv',batch_size=batch_size, shuffle=True)
    
    train_dataset = textDataset(args.train_file, train=True)

    train_loader, test_loader, vec = create_dataloader(train_dataset, args.train_file, args.val_file, args.batch_size, shuffle=True)
    input_size = len(vec.get_feature_names_out())

    model = NN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) 
    #T_max每个周期的最大epoch数，在一个周期结束后，学习率将被重置

    best_val_acc = 0 
    print('-----start train-----')
    for epoch in tqdm(range(args.num_epochs)):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        writer.add_scalar('Loss/train',train_loss,epoch)
        writer.add_scalar('Acc/train',train_acc,epoch)
        val_loss, val_accuracy = evaluate(model, test_loader, criterion)
        writer.add_scalar('Loss/val',val_loss,epoch)
        writer.add_scalar('Acc/val',val_accuracy,epoch)
        print(f'epoch [{epoch+1}/{num_epochs}], Train Loss:{train_loss:.5f}, Train Acc:{train_acc:.5f}, Val Loss:{val_loss:.5f}, Val Acc:{val_accuracy:.5f}')

        if val_accuracy > best_val_acc :#and epoch%100==0 and epoch!=0:
            torch.save(model.state_dict(), f'./checkpoint/model_epoch{epoch+1}_ValLoss{val_loss:.8f}.pth')
            print(f'model saved to model_epoch{epoch+1}_ValLoss{val_loss}.pth')
            best_val_acc = val_accuracy
            counter = 0 #早停计数
        else:
            counter += 1
            print(f'no improvement for {counter} epochs')
        
        if counter>=patience:
            print('-----early stop-----')
            break

        scheduler.step() #更新学习率
        print(f'learning rate:{scheduler.get_last_lr()}')
            
    print('-----training end-----')
    torch.save(model.state_dict(), f'./checkpoint/model_epoch{epoch+1}_ValLoss{val_loss:.8f}.pth')
    print(f'model saved to model_epoch{epoch+1}_ValLoss{val_loss}.pth')

    writer.close()

if __name__ == '__main__':
    main()