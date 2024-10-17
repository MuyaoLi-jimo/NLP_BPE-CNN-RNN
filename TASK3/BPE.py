"""
use BPE method to construct a word dictionary
"""

from pathlib import Path
from rich import print,console
from argparse import Namespace
from collections import Counter
import collections
import numpy as np
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm
import re
import utils
from tqdm import tqdm


puncs = ['。', '、', '？', '！', '『', '』', '「', '」', '…', '：', '（', '）'] + ['.', ',', '?', '!', ';', ':', '(', ')' , '<', '>', '[', ']', '...','~','"']
regex_pattern = "(" + "|".join([re.escape(punc) for punc in puncs]) + ")"

def loading_source_dataset(dataset_fold = Path(__file__).parent/"dataset"):
    dataset_path=dataset_fold / r"eng_jpn.txt"
    with open(dataset_path, mode="r",encoding="UTF-8") as file:
        lines = file.readlines()
    source_dict = {"eng":[],"jpn":[]}
    for line in lines:
        line = line.strip()
        if line == "":
            break
        source_dict["jpn"].append(line.split("\t")[0])
        source_dict["eng"].append(line.split("\t")[1])
    return source_dict


def init_word_dict(sources,lan,):

    # 使用词表进行分词
    vocab_all = []
    
    # 首先把标点符号和词语拆开，注意“'”不要分
    
    for sentence in sources:
        new_sentence = sentence.lower()
        new_sentence = re.sub(regex_pattern, r" \1 ", new_sentence)

        vocab_all.extend([" ".join(list(word)) + " _" for word in new_sentence.split()]) #注意是" _"
    word_freqs = Counter(vocab_all)

    if (Path("__file__").parent/"tokenizer"/f"{lan}_word_splits.json").exists():
        word_splits = utils.load_json_file(Path("__file__").parent/"tokenizer"/f"{lan}_word_splits.json")
        alphabet = utils.load_json_file(Path("__file__").parent/"tokenizer"/f"{lan}_alphabet.json")
        return word_freqs,word_splits,alphabet

    import copy
    word_splits = {word:copy.deepcopy(word)  for word in word_freqs.keys()}
    
    alphabet_list = []
    for voc in vocab_all:
        alphabet_list.extend(voc.split())
    alphabet = Counter(alphabet_list)
    return word_freqs,word_splits,alphabet

def compute_pair_freqs(word_freqs,word_splits):
    pair_freqs = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        split = word_splits[word].split()
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return Counter(pair_freqs)

def merge_vocab(pair,word_freqs,word_split):

    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in word_freqs.keys():
        w_in = word_split[word]
        w_out = p.sub(''.join(pair), w_in)
        word_split[word] = w_out
    return word_split

def create_tokenizer(source_dict):
    alphabet_len = {
        "eng":9998,
        "jpn":9998,
    }
    total_alphabet = {}
    for lan,source in source_dict.items():
        word_freqs,word_splits,alphabet = init_word_dict(source,lan)
        #word_splits = utils.load_json_file("word_splits.json")
        #alphabet = utils.load_json_file("alphabet.json")
        while len(alphabet)<alphabet_len[lan]:
            print(len(alphabet))
            try:
                __ = compute_pair_freqs(word_freqs,word_splits).most_common(1)[0]
            except:
                print(compute_pair_freqs(word_freqs,word_splits))
                continue
            max_pair,max_freq = __[0],__[1]
            alphabet[max_pair[0]] -= max_freq
            if alphabet[max_pair[0]]==0:
                del alphabet[max_pair[0]]
            alphabet[max_pair[1]] -= max_freq
            if alphabet[max_pair[1]]==0:
                del alphabet[max_pair[1]]
            
            alphabet[max_pair[0]+max_pair[1]] = max_freq
            
            word_splits = merge_vocab(max_pair,word_freqs,word_splits)
            if len(alphabet)%1000 == 0:
                with open(Path("__file__").parent/"tokenizer"/f"{lan}_alphabet.txt",mode="w",encoding="UTF-8") as file:
                    file.write(str(alphabet))
                utils.dump_json_file(word_splits,Path("__file__").parent/"tokenizer"/f"{lan}_word_splits.json")
                utils.dump_json_file(alphabet,Path("__file__").parent/"tokenizer"/f"{lan}_alphabet.json")
        total_alphabet.update(alphabet)    
    return total_alphabet



class MyTokenizer:
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"
    PAD = "<pad>"
    def __init__(self,word_dict_path =Path(__file__).parent/"tokenizer"/"word_dict.json", alphabet_path = Path(__file__).parent/"tokenizer"/"alphabet.json"):
        self.word_dict = {}
        if not word_dict_path.exists():
            
            alphabet = Counter(utils.load_json_file(alphabet_path))
            alphabet = Counter({key: value for key, value in alphabet.items() if value > 0})
            alphabet
            self.alphabet = alphabet
            self.max_words_len = 0
            self.word_dict[self.PAD]=0
            self.word_dict[self.UNK]=1
            self.word_dict[self.BOS]=2
            self.word_dict[self.EOS]=3
            for i,k in enumerate(self.alphabet.keys()):
                self.word_dict[k]=i+4
                if len(k)>self.max_words_len:
                    self.max_words_len = len(k)
            word_dict_raw = {
                "max_words_len":self.max_words_len,
                "word_dict":self.word_dict
                }
            utils.dump_json_file(word_dict_raw,word_dict_path)
                
        word_dict_raw = utils.load_json_file(word_dict_path)
        self.max_words_len = word_dict_raw["max_words_len"]
        self.word_dict = word_dict_raw["word_dict"]
        self.id2word = {idx:w for w,idx in self.word_dict.items() }

    @property
    def bos(self):
        return self.word_dict[self.BOS]
    
    @property
    def eos(self):
        return self.word_dict[self.EOS]

    @property
    def unk(self):
        return self.word_dict[self.UNK]
    
    @property
    def pad(self):
        return self.word_dict[self.PAD]

    def tokenize2subwords(self,sentence:str):
        """
        sentence: str
        """
        tokens = []
        new_sentence = sentence.lower()
        new_sentence = re.sub(regex_pattern, r" \1 ", new_sentence)
        words = ["".join(list(word)) + "_" for word in new_sentence.split()]
        for word in words:
            sw_tokens = []
            end_idx = min([len(word), self.max_words_len])
            start_idx = 0
            while start_idx < len(word):
                subword = word[start_idx:end_idx]
                if subword in self.word_dict:
                    sw_tokens.append(subword)
                    start_idx = end_idx
                    end_idx = min([len(word), start_idx + self.max_words_len])
                elif len(subword) == 1:
                    sw_tokens.append(self.UNK)
                    start_idx = end_idx
                    end_idx = min([len(word), start_idx + self.max_words_len])
                else:
                    end_idx -= 1
            tokens.extend(sw_tokens)
        return tokens

    def tokenize(self,sentence:str,add_tag=False,add_pad = False,max_length=32)->np.ndarray:
        # 从sentence 到 tokens
        token_ids = []
        #if add_tag:
        #    tokens.append(self.BOS)
        tokens = self.tokenize2subwords(sentence)
        if add_tag:
            tokens.append(self.EOS)
        for token in tokens:
            token_ids.append(self.word_dict[token])
        if add_pad:
            input_ids = np.zeros(max_length, dtype=np.int32)
            # 嵌入
            k = 0
            for k in range(min(max_length,len(token_ids))):
                input_ids[k] = token_ids[k]
            k+=1
            while k < max_length:
                input_ids[k] = self.word_dict[self.PAD]
                k+=1
                
        else:
            input_ids = np.array(token_ids)
        return input_ids

    def detokenize_list(self,token_ids:list):
        tokens = []
        for token_idx in token_ids:
            if token_idx == self.eos:
                break
            tokens.append( self.id2word[token_idx])
        return tokens
    
    def to_sentence(self,tokens:list):
        sentence = ""
        for token in tokens:
            sentence += token
        sentence = sentence.replace("_"," ")
        return sentence
   
    def detokenize(self,token_ids:list):
        tokens = self.detokenize_list(token_ids)
        return self.to_sentence(tokens)

if __name__ == "__main__":
    # 构建词典
    source_dataset = loading_source_dataset()
    alphabet = create_tokenizer(source_dataset)
    alphabet_path = Path("__file__").parent/"tokenizer"/"alphabet.json"
    alphabet_path.parent.mkdir(parents=True,exist_ok=True)
    utils.dump_json_file(alphabet,alphabet_path)
    # 尝试构建tokenzier
    tokenizer = MyTokenizer()
    print(tokenizer.tokenize2subwords("Hello!"))


    