import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from argparse import Namespace
import utils
from pathlib import Path
import wandb
from tqdm import tqdm
import jieba
from rich import print,console
import os
config = Namespace(
    epochs = 30,
    EMBEDDING_SIZE = 100,
    MAX_VOCAB_SIZE = 20000,
    batch_size = 64,
    lr = 1e-3,
    drop_prob=0.5,
    class_num = 4,
    seq_length = 16,
    early_stop_delta = 0,
    early_stop_patiance = 5
)

os.environ["WANDB_API_KEY"]="998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d"
wandb.login()

def criterion(out_labels,gt_labels):
    """计算loss """
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(out_labels,gt_labels)


def accuracy(out_labels, gt_labels):
    # 获取最高logit的索引，这将是预测的类别
    _, predicted = torch.max(out_labels, 1)
    
    # 检查预测的类别是否与真实标签相同
    correct = (predicted == gt_labels.squeeze()).float() # 将bool转换为float以进行计算
    
    accuracy = correct.mean()
    return accuracy.item()  

def data_prepare(data_fold = Path(__file__).parent/"dataset",data_type = "train",word_dict = utils.load_json_file(Path(__file__).parent/"word_dict.json")):
    """对输入的文件进行处理 """
    #打开文件
    data_path = data_fold / f"{data_type}.txt" 
    lines = []
    with open(data_path,mode="r",encoding="UTF-8") as file:
        lines = file.readlines()
    inputs,labels = [],[]
    # 用jieba分词
    for line in lines:
        line = line.strip()
        if line == "":
            break
        sentence,label = line.split("\t")[0],line.split("\t")[1]
        seg_list = jieba.cut(sentence, cut_all=False)
        input_sentence = np.array([config.MAX_VOCAB_SIZE-1]*config.seq_length,dtype=np.int64)

        for i,token in enumerate(seg_list):
            if i >= config.seq_length:
                break
            input_sentence[i]= word_dict.get(token,config.MAX_VOCAB_SIZE-1)
        inputs.append(input_sentence)
        labels.append(int(label))
    
    return inputs,labels

class EarlyStopping:
    def __init__(self, patience=config.early_stop_patiance, verbose=False, delta=config.early_stop_delta):
        """
        Args:
            patience (int): 有多少个epoch没有改善后停止训练.
            verbose (bool): 如果为True, 打印一条信息说明早停被触发.
            delta (float): 最小改善量,即新的loss需要比旧的loss至少低多少才算是改善.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """如果验证损失降低，则保存模型"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), Path(__file__).parent / "model" /"checkpoint.pt")
        self.val_loss_min = val_loss


class TextCNNDataset(torch.utils.data.Dataset):
    def __init__(self, data_type="train"):
        super(TextCNNDataset,self).__init__()
        self.inputs,self.labels = data_prepare(data_type=data_type)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx],dtype=torch.int64),torch.tensor(self.labels[idx],dtype=torch.long)

class TextCNN(nn.Module):
    def __init__(self,embedding_vector=None,vocab_size=config.MAX_VOCAB_SIZE,
                 embedding_dim=config.EMBEDDING_SIZE,kernel_sizes=[3, 4, 5],
                 output_channel=100,drop_prob=config.drop_prob,class_num=config.class_num):
        super(TextCNN, self).__init__()
        self.kernel_sizes = kernel_sizes
        # 1. embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if type(embedding_vector)!=type(None):
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_vector))

        # 2. convolutional layers
        self.convs_1d = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=output_channel, kernel_size=(k, embedding_dim), padding=(k-2,0)) 
            for k in kernel_sizes])

        # 3. final, fully-connected layer for classification
        self.fc = nn.Linear(len(kernel_sizes)*output_channel, class_num) 
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, x): 
        """ 
        x : [batch_size,seq_length]
        """
        embeds = self.embedding(x) # (batch_size, seq_length, embedding_dim)
        embeds = embeds.unsqueeze(1) #增加C维度，(batch_size, 1 , seq_length, embedding_dim)
        conv_x= []
        for idx in range(len(self.kernel_sizes)):
            x = F.relu(self.convs_1d[idx](embeds)).squeeze(3)  # (batch_size, num_filters,conv_seq_length)
            x_max = F.max_pool1d(x, x.size(2)).squeeze(2) #(batch_size,num_filters)
            conv_x.append(x_max)  #(3, batch_size,num_filters)
        
        x = torch.cat(conv_x, 1) #(batch_size,3*num_filters)
        
        x = self.dropout(x) 
        logit = self.fc(x)       
        return logit

def test(model:nn.Module,data_type="dev",device="cpu"):    
    dataset = TextCNNDataset(data_type=data_type)
    dataloader = torch.utils.data.DataLoader(dataset, config.batch_size, shuffle=False)
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_count = 0
    with torch.no_grad():

        for inputs, gt_labels in tqdm(dataloader):
            inputs, gt_labels = inputs.to(device), gt_labels.to(device)
            output_labels = model(inputs)
            loss = criterion(output_labels, gt_labels)
            acc = accuracy(output_labels, gt_labels)

            total_loss += loss.item() * inputs.size(0)
            total_accuracy += acc * inputs.size(0)
            total_count += inputs.size(0)

    avg_loss = total_loss / total_count
    avg_accuracy = total_accuracy / total_count
    
    return avg_loss,avg_accuracy
               

def train():
    train_on_gpu = torch.cuda.is_available()
    device = "cuda" if train_on_gpu else "cpu"
    console_0 = console.Console()

    if train_on_gpu:
        console_0.log('Training on GPU.')
    else:
        console_0.log('No GPU available, training on CPU.')

    wandb.init(project='text_cnn_training', name="10_epoch")
    wandb.config.update(vars(config))

    dataset = TextCNNDataset(data_type="train")
    dataloader = torch.utils.data.DataLoader(dataset, config.batch_size, shuffle=True)
    model = TextCNN(embedding_vector=utils.load_npy_file(Path(__file__).parent / "model" /"embedding_2.npy")).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    early_stop = EarlyStopping()

    total_train_loss = 0  # 初始化累计的训练损失
    total_samples = 0     # 累计处理的样本数量
    torch.save(model,Path(__file__).parent / "model" /"save.pt")
    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        total_samples = 0

        for i,(inputs, gt_labels) in enumerate(tqdm(dataloader)):
            inputs, gt_labels = inputs.to(device), gt_labels.to(device)
            optimizer.zero_grad()
            output_labels = model(inputs)
            loss = criterion(output_labels, gt_labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # 可以选择性地在这里记录每20步的训练损失
            if (i + 1) % 20 == 0:
                wandb.log({'Step Train Loss': loss.item()})

        avg_train_loss = total_train_loss / total_samples
        avg_val_loss, avg_val_accuracy = test(model, data_type="dev", device=device)
        
        # 在WandB和console中记录每个epoch的平均训练和验证损失及准确率
        wandb.log({
            'Epoch': epoch,
            'Average Train Loss': avg_train_loss,
            'Validation Loss': avg_val_loss,
            'Validation Accuracy': avg_val_accuracy
        })
        
        console_0.log(f'Epoch {epoch}: Average Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.2%}')
        early_stop(avg_val_loss,model=model)
        if early_stop.early_stop:
            break
    if not early_stop.early_stop:
        torch.save(model.state_dict(),Path(__file__).parent / "model" /"checkpoint.pt")

if __name__ == "__main__":
    #train()
    device="cpu"
    model = TextCNN(embedding_vector=utils.load_npy_file(Path(__file__).parent/"model" / "embedding_2.npy")).to(device)
    model.load_state_dict(torch.load(Path(__file__).parent / "model" /"checkpoint.pt",weights_only=False))
    print(test(model,data_type="test",device=device))

    


