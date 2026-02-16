import collections
import json
import regex as re
import os
import heapq
import collections
from collections import defaultdict
from typing import List, Tuple, Dict, Set
from pathlib import Path

def gpt_bytes_to_unicode_local():
    """
    将不能打印的byte转换为高位unicode字符对应二进制，能打印的保留，然后最后转成他们的字符
    """
    bs= (
        list(range(ord('!'),ord('~')+1)) # ord返回字符的unicode编码，即数字
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs=bs[:]
    n=0
    for i in range(2**8):
        if i not in bs:
            bs.append(i)
            cs.append(2**8+n)
            n+=1
    cs = [chr(n) for n in cs]
    return dict(zip(bs,cs))

def get_pair_counts(token_sequences:List[List[str]]) -> collections.Counter:
    """
    获取token序列中的所有pair的次数
    """
    pair_counts = collections.Counter()
    for sequence in token_sequences:
        for i in range(len(sequence) - 1):
            pair = (sequence[i], sequence[i + 1])
            pair_counts[pair] += 1
    return pair_counts
def merge_pair_in_sequence(sequences:List[List[str]],pair:Tuple[str,str],combined_token:str)->List[List[str]]:
    new_sequences = []
    for sequence in sequences:
        new_seq = []
        for i in range(len(sequence)):
            if i<len(sequence)-1 and (sequence[i],sequence[i+1])==pair:
                new_seq.append(combined_token)
                i+=1
            else:
                new_seq.append(sequence[i])
        new_sequences.append(new_seq)
    return new_sequences
def merge_pair_in_token_seq(token_seq:List[str],pair:Tuple[str,str],combined_token:str)->List[str]:
    new_seq = []
    i=0
    while i<len(token_seq):
        if i<len(token_seq)-1 and (token_seq[i],token_seq[i+1])==pair:
            new_seq.append(combined_token)
            i+=2
        else:
            new_seq.append(token_seq[i])
            i+=1
    return new_seq
def simple_train_bpe(input_path:str|Path,vocab_size:int,special_tokens:List[str]=None)->Tuple[Dict[int,bytes],List[Tuple[bytes,bytes]]]:
    """
    简单版，逻辑简单，时间复杂度高，无法达到要求
    Args:
        input_path:
        vocab_size:
        special_tokens:

    Returns:

    """
    _BYTE_TO_UNICODE_MAP = gpt_bytes_to_unicode_local()
    token_str_to_byte = {v:bytes([k]) for k,v in _BYTE_TO_UNICODE_MAP.items()}
    vocab:Dict[int,bytes] = {i:bytes([i]) for i in range(256)}
    next_id=256
    existing_byte_values:Set[bytes] = set(vocab.values())
    for s in special_tokens:
        if s.encode("utf-8") in existing_byte_values:
            raise ValueError(f"Special token {s} conflicts with existing byte values.")
        vocab[next_id]=s.encode()
        next_id+=1
        existing_byte_values.add(s.encode())
    try:
        with open(input_path,"r",encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        raise IOError(f"Error reading input file: {e}")
    raw_texts=re.findall(r'\s*\S+',text)
    token_sequences = []
    for raw in raw_texts:
        raw_encode= raw.encode("utf-8")
        if not raw_encode:
            continue
        token_sequence = [ _BYTE_TO_UNICODE_MAP[b] for b in raw_encode]
        token_sequences.append(token_sequence)
    merges:List[Tuple[bytes,bytes]] = []
    while len(vocab)<vocab_size:
        pair_counts=get_pair_counts(token_sequences)
        if not pair_counts:
            print("No more pairs to merge.")
            break
        most_pair=max(pair_counts,key=lambda x:pair_counts[x])
        new_most_pair_token = vocab[next_id] = token_str_to_byte[most_pair[0]] + token_str_to_byte[most_pair[1]]
        next_id+=1
        merges.append((token_str_to_byte[most_pair[0]],token_str_to_byte[most_pair[1]]))
        token_str_to_byte[new_most_pair_token.decode("utf-8")]=new_most_pair_token
        token_sequences = merge_pair_in_sequence(token_sequences,most_pair,new_most_pair_token.decode("utf-8"))
    return vocab,merges
def train_bpe(input_path:str|Path,vocab_size:int,special_tokens:List[str]=None)->Tuple[Dict[int,bytes],List[Tuple[bytes,bytes]]]:
    """

    Args:
        input_path:
        vocab_size:
        special_tokens:

    Returns:

    """
    vocab:Dict[int,bytes]={i:bytes([i]) for i in range(256)}
    next_id=256
    token_freq_dict=defaultdict(int)
    existing_byte_values:Set[bytes] = set(vocab.values())
    for s in special_tokens:
        assert s.encode("utf-8") not in existing_byte_values,f"Special token {s} conflicts with existing byte values."
        vocab[next_id]=s.encode()
        next_id+=1
        existing_byte_values.add(s.encode())
    try:
        with open(input_path,"r",encoding="utf-8") as f:
            text=f.read()
    except Exception as e:
        raise IOError(f"Error reading input file: {e}")
    chunks=re.split('|'.join(map(re.escape,special_tokens)),text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        for word in re.findall(PAT,chunk):
            word_encode=word.encode("utf-8")
            if not word_encode:
                continue
            word_encode_list=[bytes([b]) for b in word_encode]
            token_freq_dict[tuple(word_encode_list)]+=1
    merges:List[Tuple[bytes,bytes]] = []
    pair_counts=defaultdict(int)
    for token_seq,freq in token_freq_dict.items():
        for i in range(len(token_seq)-1):
            pair=(token_seq[i],token_seq[i+1])
            pair_counts[pair]+=freq
    while len(vocab)<vocab_size:
        if not pair_counts:
            print("No more pairs to merge.")
            break
        # TODO；这里的时间复杂度是O(N) N是pair的数量，考虑使用heap
        most_count=max(pair_counts.values())
        candidates=[pair for pair,count in pair_counts.items() if count==most_count]
        most_pair=max(candidates)
        merges.append(most_pair)
        new_token=most_pair[0]+most_pair[1]
        vocab[next_id]=new_token
        next_id+=1
        # TODO：更新受影响的token序列，这里的时间复杂度是O(N*M) N是token序列的数量，M是平均token序列长度，考虑使用增量更新
        affected_token_seqs = []
        for token_seq, freq in token_freq_dict.items():
            # FIXED：这里使用any需要遍历完整个token_seq，换成for循环，找到第一个匹配的pair就break，减少不必要的遍历，这是导致超时的主要问题
            # has_pair=any(token_seq[i:i+2]==list(most_pair) for i in range(len(token_seq)-1))
            has_pair=False
            for i in range(len(token_seq)-1):
                if (token_seq[i],token_seq[i+1])==most_pair:
                    has_pair=True
                    break
            if has_pair:
                affected_token_seqs.append((token_seq,freq))
        for token_seq,freq in affected_token_seqs:
            for i in range(len(token_seq)-1):
                pair_counts[(token_seq[i],token_seq[i+1])]-=freq
                if pair_counts[(token_seq[i],token_seq[i+1])]<=0:
                    del pair_counts[(token_seq[i],token_seq[i+1])]
            new_token_seq=merge_pair_in_token_seq(token_seq,most_pair,new_token)
            for i in range(len(new_token_seq)-1):
                pair_counts[(new_token_seq[i],new_token_seq[i+1])]+=freq
            # del token_freq_dict[token_seq]
            token_freq_dict[tuple(new_token_seq)]=token_freq_dict.pop(token_seq)
        byte_encoder=gpt_bytes_to_unicode_local()
        # with open("./vocab.txt","w",encoding="utf-8") as f:
        #     for idx,token in vocab.items():
        #         readable_token = "".join(byte_encoder[b] for b in token)
        #         f.write(f"{idx}\t{readable_token}\n")
        readable_vocab = {}
        for idx,token in vocab.items():
            readable_token = "".join(byte_encoder[b] for b in token)
            readable_vocab[idx] = readable_token

        with open("./vocab.json", "w", encoding="utf-8") as f:
            json.dump(readable_vocab, f, ensure_ascii=False, indent=2)
        with open("./merges.txt","w",encoding="utf-8") as f:
            for token1,token2 in merges:
                readable_token1 = "".join(byte_encoder[b] for b in token1)
                readable_token2 = "".join(byte_encoder[b] for b in token2)
                f.write(f"{readable_token1}\t{readable_token2}\n")
    return vocab,merges







