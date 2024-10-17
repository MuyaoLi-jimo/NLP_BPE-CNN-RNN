import jieba
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
from pyecharts.charts import Scatter
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
import scipy
from tqdm import tqdm

#os.environ["WANDB_API_KEY"]=
wandb.login()

np.random.seed(1)



config = Namespace(
    C = 3, # 窗口大小
    K = 5, # 负采样样本数（噪声词）
    epochs = 10,#事实证明3个epoch就差不多了
    MAX_VOCAB_SIZE = 20000,
    EMBEDDING_SIZE = 100,
    batch_size = 256,
    lr = 0.025,
    train_split = 0.95
)



def creating_dataset(dataset_fold = Path(__file__).parent/"dataset"):
    """处理原始数据集 """
    dataset_path = dataset_fold / "word2vector"/"data.json"
    part_list = ["train","test","dev"]
    jieba.load_userdict("/Users/jimoli/Desktop/my_hw/NLP assignment1/TASK2/jieba/extra_dict/dict.txt.small")
    dataset = []
    for part_name in part_list:
        
        part_file_path = dataset_fold / f"{part_name}.txt"
        with open(part_file_path,mode="r",encoding="UTF-8") as file:
            lines = file.readlines()

        line_list = []
        for line in lines:
            line = line.strip()
            if line == "":
                break
            sentence = line.split("\t")[0]
            seg_list = jieba.cut(sentence, cut_all=False)
            line_list.append([token for token in seg_list])
        dataset.extend(line_list)
    
    train_dataset = dataset[:int(len(dataset)*config.train_split)]
    val_dataset = dataset[int(len(dataset)*config.train_split):]
    utils.dump_json_file({"train":train_dataset,"val":val_dataset},dataset_path)

def append_dataset(dataset_fold = Path(__file__).parent/"dataset"):
    append_data_path = dataset_fold / "word2vector"/"renminribao.txt"
    dataset_path = dataset_fold / "word2vector"/"data.json"
    datas = utils.load_json_file(dataset_path)
    with open(append_data_path,mode="r",encoding="UTF-8") as file:
        lines = file.read()
    tokens = lines.split()
    datas["train"].append(tokens)
    utils.dump_json_file(datas,dataset_path)

def make_dataset(dataset_path = Path(__file__).parent/"dataset"/"word2vector"/"data.json"):
    pre_dataset = utils.load_json_file(dataset_path)
    dataset_list = []
    for e in pre_dataset["train"]:
        dataset_list.extend(e)
    vocab_dict = dict(Counter(dataset_list).most_common(config.MAX_VOCAB_SIZE-1))
    vocab_dict['<UNK>'] = len(dataset_list) - np.sum(list(vocab_dict.values())) # 把不常用的单词都编码为"<UNK>"
    print(vocab_dict['<UNK>'])
    # 构建词值对
    word2idx = {word:i for i, word in enumerate(vocab_dict.keys())}
    idx2word = {i:word for i, word in enumerate(vocab_dict.keys())}

    # 计算和处理频率
    word_counts = np.array(list(vocab_dict.values()), dtype=np.float32)
    word_freqs = (word_counts / np.sum(word_counts))** (3./4.) 
    utils.dump_json_file(word2idx,Path(__file__).parent/"word_dict.json")
    
    return pre_dataset,word2idx,word_freqs

class WordEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, datas, word2idx, word_freqs):
        super(WordEmbeddingDataset,self).__init__()
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)
        self.index = []
        self.datas= []
        for kdx,data in enumerate(datas):
            words_id = torch.LongTensor([word2idx.get(word, word2idx['<UNK>']) for word in data])
            self.datas.append(words_id)
            for jdx in range(config.C,len(words_id)-config.C-1):
                self.index.append((kdx,jdx))
    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx):
        kdx,jdx = self.index[idx]
        center_words = self.datas[kdx][jdx]
        pos_indices = list(range(jdx - config.C, jdx)) + list(range(jdx + 1, jdx + config.C + 1))
        pos_words = self.datas[kdx][pos_indices]
        neg_words = torch.multinomial(self.word_freqs, config.K * pos_words.shape[0], True)
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
    wandb.init(project='word2vec_training',name="10epoch")
    wandb.config.update(vars(config))
    datas,word2idx,word_freqs = make_dataset()
    dataset = WordEmbeddingDataset(datas["train"], word2idx, word_freqs)
    val_dataset = WordEmbeddingDataset(datas["val"], word2idx, word_freqs)
    dataloader = torch.utils.data.DataLoader(dataset, config.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, config.batch_size, shuffle=False)
    model = WordVec().cuda()
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
        embeddings = model.input_embedding()
        utils.dump_npy_file(embeddings,Path(__file__).parent/f"embedding_{epoch}.npy")

def find_nearest(word):
    word2idx = utils.load_json_file(Path(__file__).parent/"word_dict.json")
    idx2word = {}
    for a,b in word2idx.items():
        idx2word[b]=a
    embedding_weights=utils.load_npy_file(Path(__file__).parent/"embedding_2.npy")
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx2word[i] for i in cos_dis.argsort()[:10]]

def visualize():
    # 降维并绘制前300个词的关联散点图
    word2idx = utils.load_json_file(Path(__file__).parent/"word_dict.json")
    # TSNE降维
    embedding_weights=utils.load_npy_file(Path(__file__).parent/"embedding.npy")
    tsne = TSNE(n_components=2, learning_rate=100).fit_transform(embedding_weights)

    x_data =[]
    y_data =[]
    index = 300 # 注意我们这里为了防止数据太过密集，只取前300个词来进行绘制
    for i, label in enumerate(list(word2idx.keys())[:index]):
        x, y = float(tsne[i][0]), float(tsne[i][1])
        x_data.append(x)
        y_data.append((y, label))

    (
        Scatter(init_opts=opts.InitOpts(width="16000px", height="10000px"))
        .add_xaxis(xaxis_data=x_data)
        .add_yaxis(
            series_name="",
            y_axis=y_data,
            symbol_size=50,
            label_opts=opts.LabelOpts(
                font_size=50,
                formatter=JsCode(
                    "function(params){return params.value[2];}"
                )
            ),
        )
        .set_series_opts()
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(type_="value"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
            ),
            tooltip_opts=opts.TooltipOpts(is_show=False),
        )
        .render("scatter.html")
    )

if __name__=="__main__":
    print(find_nearest("我"))
    
    

                
