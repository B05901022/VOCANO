# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:26:13 2019

@author: Austin Hsu
"""

import torch
import torch.nn as nn
import math

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#######################################################################################
# ShakeDrop Reference
# https://github.com/owruby/shake-drop_pytorch/
#

class ShakeDropFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpharange=[-1,1]):
        if training:
            gate = torch.FloatTensor([0]).bernoulli_(1-p_drop).to(x.device)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.FloatTensor(x.size(0)).uniform_(*alpharange).to(x.device)
                alpha = alpha.view(alpha.size(0),1,1,1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1-p_drop) * x
    
    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.FloatTensor(grad_output.size(0)).uniform_(0,1).to(grad_output.device)
            beta = beta.view(beta.size(0),1,1,1).expand_as(grad_output)
            beta = torch.autograd.Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None
            
class ShakeDrop(nn.Module):
    
    def __init__(self, p_drop, alpha=[-1,1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha  = alpha
    
    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha)
