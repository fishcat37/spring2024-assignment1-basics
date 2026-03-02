import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from typing import Tuple
import math


class AdamW(Module):
    def __init__(self,params:torch.nn.ParameterList, lr:float,betas:Tuple[float,float],eps:float,weight_decay:float):
        super().__init__()
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # 初始化一阶动量和二阶动量，以及时间步长
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
    def step(self):
        self.t += 1
        beta1, beta2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            # 更新一阶动量和二阶动量
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g * g
            # 计算偏差修正后的动量，因为动量初始值为0，所以需要进行修正
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            # 更新参数，这里的更新公式来源于loss函数的梯度下降，即loss函数关于参数的导数，并且加入了权重衰减项，而这个权重衰减项是直接作用于参数更新而不是损失函数的，此处第一项的分母是归一化项，通过缩放各个参数更新步长来让优化趋向各向同性
            p.data -= self.lr * (m_hat / (torch.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)


def get_lr_cosine_schedule(it:int,max_learning_rate:float,min_learning_rate:float,warmup_iters:int,cosine_cycle_iters:int)->float:
    if it<warmup_iters:
        return max_learning_rate*it/warmup_iters
    else:
        return min_learning_rate+0.5*(max_learning_rate-min_learning_rate)*(1+math.cos(math.pi*((it-warmup_iters)%cosine_cycle_iters)/cosine_cycle_iters))