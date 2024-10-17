import numpy
from time import time
from rich.console import Console


def flatten_list_1(nested_list: list):
    """Flatten a list.
    """
    output_list = []
    for a in nested_list:
        output_list.extend(a)
    return output_list

def flatten_list_2(nested_list: list):
    """Flatten a list.
    """
    output_list = []
    for a in nested_list:
        for b in a:
            output_list.append(b)
    return output_list

def flatten_list(nested_list: list):
    """Flatten a list.
    """
    begin = time()
    output_list = flatten_list_2(nested_list)
    Console().log(f"[green]flatten_list:长度{len(output_list)}, 用时{time()-begin}")
    return output_list


def char_count_1(s: str):
    """使用dict记录 """
    out_dict = {}
    for i,_ in enumerate(s):
        if not s[i].islower():
            continue
        temp = out_dict.get(s[i],0)
        out_dict[s[i]]=temp+1
    return out_dict

def char_count_2(s: str):
    """使用list记录 """
    counts = [0] * 26
    base = ord('a')
    
    for char in s:
        if 'a' <= char <= 'z':
            # 计算字符对应的数组索引
            index = ord(char) - base
            counts[index] += 1

    out_dict = {}
    for i in range(26):
        if counts[i] > 0:
            out_dict[chr(i + base)] = counts[i]
            
    return out_dict



def char_count(s: str):
    begin = time()
    out_dict = char_count_2(s)
    Console().log(f"[green]:长度{len(s)}, char_count用时{time()-begin}")
    return out_dict