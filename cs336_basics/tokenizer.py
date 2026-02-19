from typing import List,Dict,Tuple,Iterable,Iterator,Set
import regex as re
import json
from regex import Pattern
import heapq

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
        # self.byte_encoder:Dict[int,str] = gpt_bytes_to_unicode_local()
        # self.byte_decoder:Dict[str,int] = {v:k for k,v in self.byte_encoder.items()}
        self.token_to_id:Dict[bytes,int] = {token:idx for idx,token in vocab.items()}
        self.id_to_token:Dict[int,bytes] = {idx:token for idx,token in vocab.items()}

        self.merge_dict:Dict[Tuple[bytes,bytes],int] = {(a,b): rank for rank, (a,b) in enumerate(merges)}


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
            self._special_split_re = re.compile('(' + '|'.join(escaped) + ')')
        else:
            self._special_split_re = None

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
            with open(vocab_filepath, "r", encoding="utf-8") as f:
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
                # 直接使用原始字节列表进行 BPE 合并
                word_encoded_list:List[bytes] = [bytes([b]) for b in word_encoded]
                word_encoded_heap:List[Tuple[int,Tuple[bytes,bytes]]]=[]
                for i in range(len(word_encoded_list)-1):
                    pair:Tuple[bytes,bytes]=(word_encoded_list[i],word_encoded_list[i+1])
                    if pair in self.merge_dict:
                        heapq.heappush(word_encoded_heap,(self.merge_dict.get(pair),pair))
                while word_encoded_heap:
                    rank,pair=heapq.heappop(word_encoded_heap)
                    new_word_encoded_list:List[bytes]=[]
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
                    # 直接用原始字节查表
                    tid = self.token_to_id.get(tok)
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
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding that yields token IDs one at a time.

        Args:
            iterable: An iterable of strings (e.g., lines from a file)

        Yields:
            Token IDs one at a time
        """
        for text in iterable:
            if not text:
                continue
            # Handle special tokens at the beginning if present
            # Use encode method but yield IDs one by one
            ids = self.encode(text)
            for token_id in ids:
                yield token_id
    def decode(self,ids:List[int])->str:
        """


        Args:
            ids:

        Returns:

        """
        token_bytes_list:List[bytes] = []
        for id in ids:
            token = self.id_to_token.get(id)
            if token is None:
                unk_id = self.token_to_id.get(b"<unk>")
                if unk_id is None:
                    raise ValueError(f"Token ID {id} not found in vocab and no <unk> defined.")
                token = self.id_to_token.get(unk_id)
            token_bytes_list.append(token)
        # 拼接所有 token 的原始字节
        all_bytes = b"".join(token_bytes_list)
        # 尝试用 UTF-8 解码，如果失败则替换无效序列
        return all_bytes.decode("utf-8", errors="replace")