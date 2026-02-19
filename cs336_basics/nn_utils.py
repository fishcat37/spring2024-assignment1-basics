import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from jaxtyping import Float,Bool
from einops import rearrange,repeat


class SentenceCrossEntropyLoss(Module):
    def forward(
        self,
        inputs: Float[Tensor, "b s v"],
        targets: Float[Tensor, "b s"],
        pad_idx: Bool[Tensor, "b s"]
    ) -> Float[Tensor, "()"]:
        """
        计算交叉熵损失，忽略 padding 位置。
        
        Args:
            inputs: 形状为 (batch_size, seq_len, vocab_size) 的 logits
            targets: 形状为 (batch_size, seq_len) 的目标 token ID
            pad_idx: 形状为 (batch_size, seq_len) 的布尔掩码，True 表示该位置是 padding
            
        Returns:
            标量，非 padding 位置的平均交叉熵损失
        """
        # 获取 batch_size 和 seq_len
        batch_size, seq_len, vocab_size = inputs.shape
        
        # 展平 inputs 和 targets
        # inputs: (b*s, v), targets: (b*s,)
        inputs_flat = rearrange(inputs, "b s v -> (b s) v")
        targets_flat = rearrange(targets, "b s -> (b s)")
        pad_idx_flat = rearrange(pad_idx, "b s -> (b s)")
        
        # 这里可考虑换成logsumexp，它会自动使用数值稳定的方式计算log(sum(exp(x)))，避免溢出问题
        # 数值稳定性：减去最大值 (log-sum-exp trick)
        max_logits = inputs_flat.max(dim=-1, keepdim=True).values
        logits_stable = inputs_flat - max_logits
        
        # 计算 log_softmax
        # log_softmax(x) = x - log(sum(exp(x)))
        log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True))
        log_probs = logits_stable - log_sum_exp
        
        # 使用高级索引获取目标类别的 log probability
        # 对于每个样本，获取 targets_flat[i] 位置的 log_prob
        batch_indices = torch.arange(len(targets_flat), device=inputs.device)
        target_log_probs = log_probs[batch_indices, targets_flat]
        
        # 计算交叉熵损失：-log(p[target_class])
        losses = -target_log_probs
        
        # 根据 pad_idx 过滤：pad_idx=True 表示需要忽略
        # 所以我们保留 pad_idx=False 的位置
        valid_mask = ~pad_idx_flat
        
        # 计算有效位置的平均损失
        valid_losses = losses[valid_mask]
        
        if len(valid_losses) == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        return valid_losses.mean()

class CrossEntropyLoss(Module):
    def forward(self, inputs: Float[Tensor,"b v"], targets: Float[Tensor,"b"])->Float[Tensor,"()"]:
        """
        计算交叉熵损失。
        
        Args:
            inputs: 形状为 (batch_size, vocab_size) 的 logits
            targets: 形状为 (batch_size,) 的目标 token ID
            
        Returns:
            标量，平均交叉熵损失
        """
        # 获取 batch_size 和 vocab_size
        batch_size, vocab_size = inputs.shape
        
        # 数值稳定性：减去最大值 (log-sum-exp trick)
        max_logits = inputs.max(dim=-1, keepdim=True).values
        logits_stable = inputs - max_logits
        
        # 计算 log_softmax
        log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True))
        log_probs = logits_stable - log_sum_exp
        
        # 使用高级索引获取目标类别的 log probability
        batch_indices = torch.arange(batch_size, device=inputs.device)
        target_log_probs = log_probs[batch_indices, targets]
        
        # 计算交叉熵损失：-log(p[target_class])
        losses = -target_log_probs
        
        return losses.mean()

class Softmax(Module):
    def __init__(self):
        super().__init__()
    def forward(self,x:Float[Tensor,"b s v"])->Float[Tensor,"b s v"]:
        max_x,_=x.max(dim=-1,keepdim=True)
        x_stable=x-max_x
        exp_x=torch.exp(x_stable)
        sum_exp_x=exp_x.sum(dim=-1,keepdim=True)
        return exp_x/sum_exp_x
class GradientClipping(Module):
    def __init__(self, max_norm: float):
        super().__init__()
        self.max_norm = max_norm
    def forward(self, parameters):
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > self.max_norm:
            clip_coef = self.max_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)