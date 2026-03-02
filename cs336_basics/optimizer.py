import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from typing import Tuple
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self,params:torch.nn.ParameterList, lr:float,betas:Tuple[float,float],eps:float,weight_decay:float):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad() # 优化过程不需要梯度，建议显式加上
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # 状态初始化（延迟初始化，确保设备一致）
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p) # 一阶动量
                    state['exp_avg_sq'] = torch.zeros_like(p) # 二阶动量

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # 1. 权重衰减 (Decoupled Weight Decay)
                p.mul_(1 - lr * weight_decay)
                # 2. 更新动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 3. 偏置修正
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                # 4. 更新参数,这里的更新公式来源于loss函数的梯度下降，即loss函数关于参数的导数
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


def get_lr_cosine_schedule(it:int,max_learning_rate:float,min_learning_rate:float,warmup_iters:int,cosine_cycle_iters:int)->float:
    if it<warmup_iters:
        return max_learning_rate*it/warmup_iters
    # 因为这里的测试要求不进行周期，在cosine_cycle_iters之后保持min_learning_rate，所以直接返回min_learning_rate即可
    # else:
    #     return min_learning_rate+0.5*(max_learning_rate-min_learning_rate)*(1+math.cos(math.pi*((it-warmup_iters)%(cosine_cycle_iters-warmup_iters))/(cosine_cycle_iters-warmup_iters)))
    elif it<cosine_cycle_iters:
        return min_learning_rate+0.5*(max_learning_rate-min_learning_rate)*(1+math.cos(math.pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)))
    else:
        return min_learning_rate