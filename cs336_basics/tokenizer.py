from typing import List,Dict,Tuple,Iterable,Iterator,Set
import regex as re
import json
from regex import Pattern
import heapq

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

class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    def __init__(self,vocab:Dict[int,bytes],merges:List[Tuple[bytes,bytes]],special_tokens:List[str]=None):
        """

        Args:
            vocab:
            merges:
            special_tokens:
        """
        self.vocab:Dict[int,bytes] = vocab
        self.merges:List[Tuple[bytes,bytes]] = merges
        self.byte_encoder:Dict[int,str] = gpt_bytes_to_unicode_local()
        self.byte_decoder:Dict[str,int] = {v:k for k,v in self.byte_encoder.items()}
        self.token_to_id:Dict[bytes,int] = {token:idx for idx,token in vocab.items()}
        self.id_to_token:Dict[int,bytes] = {idx:token for idx,token in vocab.items()}
        self.merge_dict:Dict[Tuple[str,str],int] = {}
        for rank, (a, b) in enumerate(merges):
            self.merge_dict[(a.decode("utf-8"), b.decode("utf-8"))] = rank
        self.special_tokens:List[str] = special_tokens if special_tokens else []
        self.special_token_set:Set[str] = set(self.special_tokens)
        self.special_token_to_id:Dict[str,int] = {}
        for s in self.special_tokens:
            sid = self.token_to_id.get(s.encode('utf-8'))
            if sid is not None:
                self.special_token_to_id[s] = sid
            else:
                raise ValueError(f"Special token {s} not found in vocab.")
        if self.special_tokens:
            escaped:List[str] = [re.escape(tok) for tok in sorted(self.special_tokens,key=len,reverse=True)]
            self._special_split_re:Pattern = re.compile('(' + '|'.join(escaped) + ')')
        else:
            self._special_split_re:Pattern = None
        
    @classmethod
    def from_files(cls,vocab_filepath:str,merges_filepath:str,special_tokens:List[str]=None):
        """

        Args:
            vocab_filepath:
            merges_filepath:
            special_tokens:

        Returns:

        """
        vocab={}
        # try:
        #     with open(vocab_filepath,'r',encoding='utf-8') as f:
        #         for line in f:
        #             idx,token=line.strip().split('\t')
        #             vocab[int(idx)]=token.encode('utf-8')
        # except Exception as e:
        #     raise IOError(f"Error reading vocab file: {e}")
        try:
            with open("./vocab.json", "r", encoding="utf-8") as f:
                readable_vocab = json.load(f)
                vocab = {int(idx): token.encode("utf-8") for idx, token in readable_vocab.items()}
        except Exception as e:
            raise IOError(f"Error reading vocab file: {e}")
        merges=[]
        try:
            with open(merges_filepath,'r',encoding='utf-8') as f:
                for line in f:
                    token1,token2=line.strip().split()
                    merges.append((token1.encode('utf-8'),token2.encode('utf-8')))
        except Exception as e:
            raise IOError(f"Error reading merges file: {e}")
        return cls(vocab,merges,special_tokens)
    def encode(self,text:str)->List[int]:
        """简单版

        Args:
            text (str): 需要进行分词的文本

        Returns:
            List[int]: token_ids
        """
        parts = [text]
        if self._special_split_re is not None:
            parts =[p for p in self._special_split_re.split(text) if p]
        ids:List[int] = []
        for part in parts:
            if part in self.special_token_set:
                sid = self.special_token_to_id.get(part)
                if sid is None:
                    unk_id = self.token_to_id.get(b"<unk>")
                    if unk_id is None:
                        raise ValueError(f"Special token {part!r} not in vocab and no <unk> defined.")
                    sid = unk_id
                ids.append(sid)
                continue
            words = re.findall(self.PAT,part)
            for word in words:
                word_encoded:bytes = word.encode("utf-8")
                word_encoded_list:List[str] = [self.byte_encoder[b] for b in word_encoded]
                word_encoded_heap:List[Tuple[int,Tuple[str,str]]]=[]
                for i in range(len(word_encoded_list)-1):
                    pair:Tuple[str,str]=(word_encoded_list[i],word_encoded_list[i+1])
                    if pair in self.merge_dict:
                        heapq.heappush(word_encoded_heap,(self.merge_dict.get(pair),pair))
                while word_encoded_heap:
                    rank,pair=heapq.heappop(word_encoded_heap)
                    new_word_encoded_list:List[str]=[]
                    i=0
                    while i<len(word_encoded_list):
                        if i<len(word_encoded_list)-1 and (word_encoded_list[i],word_encoded_list[i+1])==pair:
                            new_word_encoded_list.append(pair[0]+pair[1])
                            i+=2
                        else:
                            new_word_encoded_list.append(word_encoded_list[i])
                            i+=1
                    word_encoded_heap=[]
                    for i in range(len(new_word_encoded_list)-1):
                        pair=(new_word_encoded_list[i],new_word_encoded_list[i+1])
                        if pair in self.merge_dict:
                            heapq.heappush(word_encoded_heap,(self.merge_dict.get(pair),pair))
                    word_encoded_list=new_word_encoded_list
                unk_id = self.token_to_id.get(b"<unk>")
                for tok in word_encoded_list:
                    tid = self.token_to_id.get(tok.encode("utf-8"))
                    if tid is None:
                        if unk_id is None:
                            raise ValueError(f"Token {tok!r} not found in vocab and no <unk> defined.")
                        tid = unk_id
                    ids.append(tid)
        return ids
    
    def better_encode(self,text:str)->List[int]:
        """

        Args:
            text:

        Returns:

        """
        pass
    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        """

        Args:
            iterable:

        Returns:

        """
        pass
    def decode(self,ids:List[int])->str:
        """

        Args:
            ids:

        Returns:

        """
        token_list:List[bytes] = []
        for id in ids:
            token = self.id_to_token.get(id)
            if token is None:
                unk_id = self.token_to_id.get(b"<unk>")
                if unk_id is None:
                    raise ValueError(f"Token ID {id} not found in vocab and no <unk> defined.")
                token = self.id_to_token.get(unk_id)
            token_list.append(token)
        token_bytes = b"".join(token_list)
        token_str = token_bytes.decode("utf-8")
        decoded_tokens:List[str] = []
        for char in token_str:
            if char in self.byte_decoder:
                decoded_tokens.append(bytes([self.byte_decoder[char]]))
            else:
                decoded_tokens.append(bytes([ord(char)]))
        return b"".join(decoded_tokens).decode("utf-8")