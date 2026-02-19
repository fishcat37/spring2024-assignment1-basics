import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from jaxtyping import Float
from einops import rearrange,repeat

class GeLU(nn.Module):
    
    def forward(self,x:Float[Tensor,"... d"])->Float[Tensor,"... d"]:
        return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    
class RMSNorm(Module):
    """RMSNorm
    gamma是一个可学习参数，初始化为1，维度和输入的最后一维相同
    """
    def __init__(self,d_model:int,eps:float=1e-5):
        super().__init__()
        self.eps=eps
        self.gamma:Float[Tensor,"d"]=nn.Parameter(torch.ones(d_model))
    def forward(self,x:Float[Tensor,"... d"])->Float[Tensor,"... d"]:
        rms:Float[Tensor,"... d"] = torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        return x/rms*self.gamma
class PositionWiseFeedforward(Module):
    def __init__(self,d_model:int,d_ff:int,pdrop:float=0.1,bias:bool=False):
        super().__init__()
        self.w1=nn.Linear(d_model,d_ff,bias=bias)
        self.w2=nn.Linear(d_ff,d_model,bias=bias)
        self.gelu=GeLU()
        self.dropout=nn.Dropout(pdrop)
    def forward(self,x:Float[Tensor,"... d"])->Float[Tensor,"... d"]:
        return self.w2(self.dropout(self.gelu(self.w1(x))))
class ScaledDotProductAttention(Module):
    def __init__(self,pdrop:float=0.1):
        super().__init__()
        self.dropout=nn.Dropout(pdrop)
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,q:Float[Tensor,"... h s_q d_k"],k:Float[Tensor,"... h s_k d_k"],v:Float[Tensor,"... h s_k d_v"],mask:Float[Tensor,"... h s_q s_k"]=None)->Float[Tensor,"... h s_q d_v"]:
        d_k=q.size(-1)
        # 此处使用einops表示更加清晰，但是会copy，考虑性能可以使用transpose来实现
        score=torch.matmul(q,rearrange(k,"... h s_k d_k->... h d_k s_k"))/torch.sqrt(torch.tensor(d_k,dtype=torch.float32))
        if mask is not None:
            score=score.masked_fill(mask,float('-inf'))
        attn=self.softmax(score)
        attn=self.dropout(attn)
        return torch.matmul(attn,v)
class MutiHeadSelfAttention(Module):
    def __init__(self,d_model:int,n_heads:int,pdrop:float=0.1,bias:bool=False):
        super().__init__()
        self.qproj=nn.Linear(d_model,d_model,bias=bias)
        self.kproj=nn.Linear(d_model,d_model,bias=bias)
        self.vproj=nn.Linear(d_model,d_model,bias=bias)
        self.out_proj=nn.Linear(d_model,d_model,bias=bias)
        self.attention=ScaledDotProductAttention(pdrop)
        self.n_heads=n_heads
        self.d_k=d_model//n_heads
        self.dropout=nn.Dropout(pdrop)
    def forward(self,x:Float[Tensor,"b s d"],mask:Float[Tensor,"b s_1 s_2"]=None)->Float[Tensor,"b s d"]:
        # 使用einops，表示更加清晰，但是会copy，考虑性能可以使用view和transpose来实现
        q=rearrange(self.qproj(x),"b s (h d)->b h s d",h=self.n_heads)
        k=rearrange(self.kproj(x),"b s (h d)->b h s d",h=self.n_heads)
        v=rearrange(self.vproj(x),"b s (h d)->b h s d",h=self.n_heads)
        if mask is not None:
            mask=rearrange(mask,"b s_1 s_2->b 1 s_1 s_2")
        else:
            mask=repeat(torch.triu(torch.ones(x.size(1),x.size(1),device=x.device),diagonal=1).bool(),"s_1 s_2->b 1 s_1 s_2",b=x.size(0))
        attn=self.attention(q,k,v,mask)
        attn=rearrange(attn,"b h s d->b s (h d)")
        return self.dropout(self.out_proj(attn))


class TransformerBlock(Module):
    def __init__(self,d_model:int,n_heads:int,d_ff:int,pdrop:float=0.1,bias:bool=False):
        super().__init__()
        self.attention=MutiHeadSelfAttention(d_model,n_heads,pdrop,bias)
        self.norm1=RMSNorm(d_model)
        self.ffn=PositionWiseFeedforward(d_model,d_ff,pdrop,bias)
        self.norm2=RMSNorm(d_model)
    def forward(self,x:Float[Tensor,"b s d"],mask:Float[Tensor,"b s_1 s_2"]=None)->Float[Tensor,"b s d"]:
        original_x=x
        x=self.norm1(x)
        if mask is None:
            mask=repeat(torch.triu(torch.ones(x.size(1),x.size(1),device=x.device),diagonal=1).bool(),"s_1 s_2->b s_1 s_2",b=x.size(0))
        x=self.attention(x,mask)
        x=original_x+x
        original_x=x
        x=self.norm2(x)
        x=self.ffn(x)
        x=original_x+x
        return x
class Transformer(Module):
    def __init__(self,seq_len:int,vocab_size:int,d_model:int,n_heads:int,d_ff:int,n_layers:int,pdrop:float=0.1,bias:bool=False):
        super().__init__()
        self.word_embedding=nn.Embedding(vocab_size,d_model)
        self.position_embedding=nn.Embedding(seq_len,d_model)
        self.layers=nn.ModuleList([TransformerBlock(d_model,n_heads,d_ff,pdrop,bias) for _ in range(n_layers)])
        self.norm=RMSNorm(d_model)
        self.out_proj=nn.Linear(d_model,vocab_size,bias=bias)
    def forward(self,x:Float[Tensor,"b s"],mask:Float[Tensor,"b s_1 s_2"]=None)->Float[Tensor,"b s d"]:
        b,s=x.size()
        position_ids=torch.arange(s,device=x.device).unsqueeze(0).expand(b,-1)
        x=self.word_embedding(x)+self.position_embedding(position_ids)
        if mask is None:
            mask=rearrange(torch.triu(torch.ones(s,s,device=x.device),diagonal=1).bool(),"s_1 s_2->1 s_1 s_2")
        for layer in self.layers:
            x=layer(x,mask)
        return self.out_proj(self.norm(x))