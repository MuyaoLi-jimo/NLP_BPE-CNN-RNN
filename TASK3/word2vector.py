"""
生成一个分词器 
"""

from pathlib import Path
from rich import print,console
from argparse import Namespace
from collections import Counter
import torch.utils
import torch.utils.data
import utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import os
from sklearn.manifold import TSNE
import numpy as np
import scipy
from tqdm import tqdm
import BPE

os.environ["WANDB_API_KEY"]="998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d"
wandb.login()

np.random.seed(1)

config = Namespace(
    VALID_SPLIT = 0.1, #总数据集中0.1给valid
    TEST_SPLIT = 0.1, #总数据集中0.1给test
    C = 2, # 窗口大小
    K = 5, # 负采样样本数（噪声词）
    epochs = 30,#事实证明3个epoch就差不多了
    MAX_VOCAB_SIZE = 19950, 
    EMBEDDING_SIZE = 100,
    batch_size = 250,
    lr = 0.025,
    train_split = 0.95,
    early_stop_delta = 0,
    early_stop_patiance = 5
)


def create_raw_dataset(dataset_fold = Path(__file__).parent/"dataset"):
    """处理原始数据集 """
    
    if not (dataset_fold /"test.json").exists():
        source_dataset = BPE.loading_source_dataset(dataset_fold)
        dataset_length = len(list(source_dataset.values())[0])
        train_length = int(dataset_length*(1-config.VALID_SPLIT-config.TEST_SPLIT))
        valid_length = int(dataset_length*config.VALID_SPLIT)
        sample_list = np.random.permutation(dataset_length)

        train_dict = {k: [v[i] for i in sample_list[:train_length]] for k, v in source_dataset.items()}
        valid_dict = {k: [v[i] for i in sample_list[train_length:train_length+valid_length]] for k, v in source_dataset.items()}
        test_dict = {k: [v[i] for i in sample_list[train_length+valid_length:]] for k, v in source_dataset.items()}
        
        utils.dump_json_file(train_dict,dataset_fold/"train.json")
        utils.dump_json_file(valid_dict,dataset_fold/"valid.json")
        utils.dump_json_file(test_dict,dataset_fold/"test.json")

    train_dict = utils.load_json_file(dataset_fold/"train.json")
    valid_dict = utils.load_json_file(dataset_fold/"valid.json")
    test_dict = utils.load_json_file(dataset_fold/"test.json")
    return train_dict,valid_dict,test_dict

def make_dataset(raw_dataset:dict):
    """这个时候日英的训练只能分开，因为没有放到一起的数据"""
    word_freqs = {}
    word_lists = {}
    tokenizer = BPE.MyTokenizer()
    for lan,dataset_lists in raw_dataset.items():
        dataset_list = []
        word_lists[lan] = []
        for dataset_l in dataset_lists:
            dataset_ids = tokenizer.tokenize(dataset_l,add_tag=True)  #加不加？开头和结尾？
            dataset_list.extend(dataset_ids)
            word_lists[lan].append(dataset_ids)
        vocab_dict = Counter(dataset_list)
        # 计算和处理频率
        word_counts = np.zeros(config.MAX_VOCAB_SIZE,dtype=np.float32)
        for vocab_idx,times in vocab_dict.items():
            word_counts[vocab_idx] = times
        word_freqs[lan] = torch.tensor(word_counts / np.sum(word_counts))** (3./4.) 
    return word_freqs,word_lists

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
        self.epoch = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model,epoch):
        score = -val_loss
        self.epoch = epoch
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
        embeddings = model.input_embedding()
        utils.dump_npy_file(embeddings,Path(__file__).parent/"model"/f"embedding_{self.epoch}.npy")
        self.val_loss_min = val_loss

class WordEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, raw_datas):
        super(WordEmbeddingDataset,self).__init__()
        self.word_freqs,word_lists = make_dataset(raw_datas)
        self.datas = []
        self.index = []
        bias = 0
        for lan, lan_datas in word_lists.items():
            for kdx,len_datas in enumerate(lan_datas):
                words_id = torch.LongTensor(len_datas) 
                self.datas.append(words_id)
                for jdx in range(config.C,len(words_id)-config.C-1):
                    self.index.append((lan,bias+kdx,jdx))
            bias = len(self.datas)
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        lan,kdx,jdx = self.index[idx]
        center_words = self.datas[kdx][jdx]
        pos_indices = list(range(jdx - config.C, jdx)) + list(range(jdx + 1, jdx + config.C + 1))
        pos_words = self.datas[kdx][pos_indices]
        
        neg_words = torch.multinomial(self.word_freqs[lan], config.K * pos_words.shape[0], True)
        return center_words, pos_words, neg_words

class WordVec(nn.Module):
    def __init__(self):
        super(WordVec, self).__init__()
        self.vocab_size = config.MAX_VOCAB_SIZE
        self.embed_size = config.EMBEDDING_SIZE

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
    
    def forward(self, input_labels, pos_labels, neg_labels):
        # Embeddings
        input_embedding = self.in_embed(input_labels).unsqueeze(2)  # [batch_size, embed_size, 1]
        pos_embedding = self.out_embed(pos_labels)                  # [batch_size, window_size * 2, embed_size]
        neg_embedding = self.out_embed(neg_labels)                  # [batch_size, window_size * 2 * K, embed_size]
        
        # Scores
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)  # [batch_size, window_size * 2]
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze(2) # [batch_size, window_size * 2 * K]
        log_pos = F.logsigmoid(pos_dot).sum(1)  # [batch_size]
        log_neg = F.logsigmoid(neg_dot).sum(1)  # [batch_size]
        
        # Total loss
        loss = -(log_pos + log_neg)  # [batch_size]
        
        return loss
    
    def input_embedding(self):
        return self.in_embed.weight.detach().cpu().numpy()
    
def train():
    wandb.init(project='word2vec_training',name="eng_jpn")
    wandb.config.update(vars(config))
    train_data,valid_data,test_data = create_raw_dataset()
    dataset = WordEmbeddingDataset(train_data)
    val_dataset = WordEmbeddingDataset(valid_data)
    dataloader = torch.utils.data.DataLoader(dataset, config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, config.batch_size, shuffle=False)
    model = WordVec().cuda()
    early_stop = EarlyStopping()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    console.Console().log(f"[red]总共{len(dataset)}个元素")
    for epoch in range(config.epochs):
        model.train()
        for i, (input_labels, pos_labels, neg_labels) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            loss = model(input_labels.cuda(), pos_labels.cuda(), neg_labels.cuda()).mean()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                wandb.log({'Train Loss': loss.item()})
        model.eval()
        with torch.no_grad():
            val_losses = []
            for input_labels, pos_labels, neg_labels in val_dataloader:
                val_loss = model(input_labels.cuda(), pos_labels.cuda(), neg_labels.cuda()).mean()
                val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            wandb.log({'Validation Loss': avg_val_loss})
        
        print(f'Epoch {epoch}: Avg Validation Loss: {avg_val_loss}')
        early_stop(avg_val_loss,model=model,epoch=epoch)
        if early_stop.early_stop:
            break
    if not early_stop.early_stop:
        early_stop.save_checkpoint()
        

def most_similar(word):
    word2idx = utils.load_json_file(Path(__file__).parent/"word_dict.json")
    idx2word = {}
    for a,b in word2idx.items():
        idx2word[b]=a
    embedding_weights=utils.load_npy_file(Path(__file__).parent/"model"/"embedding_2.npy")
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx2word[i] for i in cos_dis.argsort()[:10]]

def similarity(word1,word2):
    word2idx = utils.load_json_file(Path(__file__).parent/"word_dict.json")
    idx2word = {}
    for a,b in word2idx.items():
        idx2word[b]=a
    embedding_weights=utils.load_npy_file(Path(__file__).parent/"model"/"embedding_2.npy")
    index1 = word2idx[word1]
    index2 = word2idx[word2]
    embedding1 = embedding_weights[index1]
    embedding2 = embedding_weights[index2]
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity

if __name__=="__main__":
    train_data,valid_data,test_data = create_raw_dataset()
    train()

    

                
