""" 
Train and testing a LSTM with Attention 
"""

import os
import wandb
from argparse import Namespace
from pathlib import Path
import utils,BPE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from rich import console
from tqdm import tqdm
import numpy as np

from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction

os.environ["WANDB_API_KEY"]="998c5dff7e8a2d9fb877a2492f1d9ac46aeda17d"
wandb.login()

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Namespace(
    epochs = 200,
    EMBEDDING_SIZE = 100,
    MAX_VOCAB_SIZE = 19950, 
    RNN_ENCODER_LAYER = 2,
    RNN_ENCODER_BIDIRECTION =True,
    batch_size = 128,
    lr = 1e-3,
    drop_prob=0.1,
    seq_length = 50,
    early_stop_delta = 0,
    early_stop_patiance = 5
)

class TransalteDataset(Dataset):
    """from eng2jpn """
    def __init__(self,split_label="train",dataset_folder=Path(__file__).parent/"dataset",reverse=False):
        """
        reverse: 是否从jpn-eng
        """
        dataset_path = dataset_folder/ f"{split_label}.json"
        self.raw_dataset = utils.load_json_file(dataset_path)
        tokenizer = BPE.MyTokenizer()

        # reverse=False: 从eng到jpn
        self.lans = ["jpn","eng"]
        if reverse:
            self.lans = reversed(self.lans)
        
        self.datas = {lan:[] for lan in self.lans}
        self.max_len = 0
        for lan ,v in self.raw_dataset.items():
            for data in v:
                input_ids = tokenizer.tokenize(data,add_tag=True,add_pad=True,max_length=config.seq_length)
                #if self.max_len<len(input_ids):
                    #self.max_len = len(input_ids)
                self.datas[lan].append(torch.LongTensor(input_ids))
            #print(self.max_len) #max=50

    def __len__(self,):
        return len(list(self.datas.values())[0])
    
    def __getitem__(self,idx):
        assert len(self.lans) == 2
        tran_pair = []
        idx_pair = []
        for lan in self.lans:
            idx_pair.append(self.datas[lan][idx])
            tran_pair.append(self.raw_dataset[lan][idx])
        return idx_pair[0],idx_pair[1],tran_pair

class EncoderRNN(nn.Module):
    def __init__(self, embedding_vector=None, input_size=config.MAX_VOCAB_SIZE, hidden_size=config.EMBEDDING_SIZE, dropout_p=config.drop_prob,num_layers=config.RNN_ENCODER_LAYER):
        super(EncoderRNN, self).__init__()
        
        self.bidirectional = config.RNN_ENCODER_BIDIRECTION
        self.num_layers =num_layers

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size) 
        if type(embedding_vector)!=type(None):
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_vector)) 
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=self.num_layers,bidirectional=self.bidirectional ,batch_first=True) 
        self.fc = nn.Linear(hidden_size * (self.bidirectional+1), hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        #(batch_size,seq)
        embedded = self.dropout(self.embedding(input))   #(batch_size,seq,hidden_size)
        output, (h_n, c_n) = self.lstm(embedded)         # (batch_size,seq,hidden_size*(bidirectional+1)),((num_layers*(bidirectional+1)),batch_size,hidden_size)
        output = self.fc(output)                         # (batch_size, seq_length, hidden_size)
        out_h_n = torch.sum(h_n, dim=0).unsqueeze(0)                 # (1,batch_size, hidden_size)

        return output, out_h_n

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size=config.EMBEDDING_SIZE*2):

        super(AttentionLayer, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size) 
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))  
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size=config.EMBEDDING_SIZE):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_vector=None,tokenizer=None,hidden_size=config.EMBEDDING_SIZE, output_size=config.MAX_VOCAB_SIZE, dropout_p=0.1):
        
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        if not tokenizer:
            self.tokenizer = BPE.MyTokenizer()
        self.embedding = nn.Embedding(output_size, hidden_size)
        if type(embedding_vector)!=type(None):
            self.embedding.weight = nn.Parameter(torch.from_numpy(embedding_vector)) 
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(hidden_size*2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.tokenizer.bos).to(device) #batchsize,1
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(config.seq_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1) 
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input)) # (batch_size,1,embedding)
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs) # (batch_size,1,embedding)

        input_lstm = torch.cat((embedded, context), dim=2) # (batch_size,1,embedding*2)
        c_0 = torch.randn(1,input.shape[0], self.hidden_size).to(device)
        output, (hidden,c_1) = self.lstm(input_lstm, (hidden,c_0))
        output = self.out(output)

        return output, hidden, attn_weights
    
class AttnRNN(nn.Module):

    def __init__(self,embedding_vector=None, vocab_size=config.MAX_VOCAB_SIZE, hidden_size=config.EMBEDDING_SIZE, dropout_p=config.drop_prob,num_layers=config.RNN_ENCODER_LAYER):
        super(AttnRNN, self).__init__()
        self.tokenizer = BPE.MyTokenizer()
        self.encoder = EncoderRNN(embedding_vector=embedding_vector,input_size=vocab_size,hidden_size=hidden_size,dropout_p=dropout_p,num_layers=num_layers)
        self.decoder = AttnDecoderRNN(embedding_vector=embedding_vector,tokenizer=self.tokenizer,hidden_size=hidden_size,output_size=vocab_size,dropout_p=dropout_p)
    
    def forward(self,input,target_tensor=None):
        encoder_outputs, encoder_hidden = self.encoder(input)
        decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target_tensor)
        return decoder_outputs

    def inference(self,sentence):
        inputs = torch.LongTensor(self.tokenizer.tokenize(sentence,add_tag=True,add_pad=True,max_length=config.seq_length))
        inputs =  inputs.unsqueeze(0).to(device)
        output = self.forward(inputs)
        output_words = sample_token(output).squeeze(0)
        translate_sentence = self.tokenizer.detokenize(output_words.tolist())
        return translate_sentence

criterion = nn.NLLLoss()  

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


def calculate_bleu(reference, candidate):
    """
    reference: list of sentence
    candidate: sentence
    """
    #print(f"{reference},  {candidate}")
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, smoothing_function=smoothie,weights=(0.25,0.25,0.25,0.25))

def sample_token(word_embeddings:torch.Tensor):
    _, topi = word_embeddings.topk(1)
    token_id = topi.squeeze(-1)
    return token_id

def valid(model:nn.Module,split_label="valid"):
    dataset = TransalteDataset(split_label=split_label)
    dataloader = torch.utils.data.DataLoader(dataset, config.batch_size, shuffle=False)
    tokenizer = BPE.MyTokenizer()
    model.eval()
    total_loss = 0
    total_bleu = 0
    count=0
    for j,(input_tensor, target_tensor,tran_pairs) in tqdm(enumerate(dataloader)):
        input_tensor,target_tensor = input_tensor.to(device),target_tensor.to(device)
        decoder_outputs = model(input_tensor,target_tensor)
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        total_loss += loss.item()
        
        for iddx in range(len(tran_pairs[0])):
            reference = tran_pairs[1][iddx]
            token_idx = sample_token(decoder_outputs[iddx])
            sentence_list  = tokenizer.detokenize_list(token_idx.tolist())
            reference_list = tokenizer.tokenize2subwords(reference)
            bleu_score = calculate_bleu([reference_list],sentence_list)
            total_bleu+=bleu_score
            count+=1

    CE_Loss = total_loss/len(dataloader)
    BLUE_score = total_bleu/count
    PPL_score = np.exp(CE_Loss)
    if split_label=="test":
        test(model)
    return CE_Loss,BLUE_score,PPL_score

def train():
    wandb.init(project='rnn_training', name="test")
    wandb.config.update(vars(config))
    console_0 = console.Console()

    model = AttnRNN(embedding_vector=utils.load_npy_file(Path(__file__).parent/"model"/"embedding_27.npy")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    dataset = TransalteDataset(split_label="train")
    dataloader = DataLoader(dataset=dataset,batch_size=config.batch_size,shuffle=True)
    early_stop = EarlyStopping()
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for i,(input_tensor, target_tensor,tran_pairs) in tqdm(enumerate(dataloader)):

            input_tensor,target_tensor = input_tensor.to(device),target_tensor.to(device)
            optimizer.zero_grad()
            decoder_outputs = model(input_tensor,target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        wandb.log({'Epoch Train Loss': total_loss / len(dataloader)})
        console_0.log(f"[red]train epoch: {epoch} is finished, loss{total_loss / len(dataloader)}")
        valid_loss,valid_BLEU_score,valid_PPL_score = valid(model)
        wandb.log({"Valid Loss":valid_loss,"Valid BLEU Score":valid_BLEU_score,"Valid PPL Score":valid_PPL_score})
        console_0.log(str({'Valid Loss':valid_loss,'Valid BLEU Score':valid_BLEU_score,'Valid PPL Score':valid_PPL_score}))
        early_stop(valid_loss,model)
        if early_stop.early_stop:
            early_stop.save_checkpoint(valid_loss,model)
            break

    if not early_stop.early_stop:
        torch.save(model.state_dict(),Path(__file__).parent / "model" /"checkpoint.pt")

def test(model:nn.Module):
    examples = ["私の名前は愛です。",
     "昨日はお肉を食べません。",
     "いただきますよう。",
     "秋は好きです。",
     "おはようございます。"]
    model.eval()
    for example in examples:
        translate_sentence = model.inference(example)
        print(f"jpn: [[{example}]] \n eng: [[{translate_sentence}]] ")

if __name__ == "__main__":
    #dataset = TransalteDataset(split_label="valid")
    #dataloader = torch.utils.data.DataLoader(dataset, config.batch_size, shuffle=False)
    #for input_tensor, target_tensor,tran_pairs in tqdm(dataloader):
        #print(tran_pairs)
        #break
    #train()
    train()
    model = AttnRNN(embedding_vector=utils.load_npy_file(Path(__file__).parent/"model"/"embedding_27.npy")).to(device).eval()
    print(model)
    print(model.encoder)
    print(model.decoder)
    exit()
    model.load_state_dict(torch.load(Path(__file__).parent/"model"/"checkpoint copy.pt",weights_only=False))
    _,bleu,ppl =valid(model,split_label="train")
    print(f"train dataset ppl:{ppl},bleu:{bleu}")    
    _,bleu,ppl =valid(model,split_label="valid")
    print(f"valida dataset ppl:{ppl},bleu:{bleu}")
    _,bleu,ppl =valid(model,split_label="test")
    print(f"test dataset ppl:{ppl},bleu:{bleu}")

