#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 09:18:32 2019

@author: assiene
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleMemoryHop(nn.Module):
    
    def __init__(self, dictionnary_size, embedding_size, is_final_layer=False):
        
        super(SingleMemoryHop, self).__init__()
        self.memory_embedding = nn.Embedding(dictionnary_size, embedding_size)
        self.output_embedding = nn.Embedding(dictionnary_size, embedding_size)
        self.is_final_layer = is_final_layer
        
        if is_final_layer:
            self.final_affine_transformation = nn.Linear(embedding_size, dictionnary_size)
            
    def set_output_embedding(self, embedding):
        self.output_embedding = embedding
        
    def set_memory_embedding(self, embedding):
        self.memory_embedding = embedding
        
    def set_linear_transformation(self, linear_transformation):
        self.final_affine_transformation = linear_transformation
        
    def forward(self, x, u):
        m = self.memory_embedding(x).sum(dim=2)
        c = self.output_embedding(x).sum(dim=2)
        
        p = F.softmax(u.bmm(m.transpose(1,2)))
        
        o = (p.transpose(1,2) * c).sum(dim=1).unsqueeze(dim=1)
        
        if self.is_final_layer:
            a_hat = F.log_softmax(self.final_affine_transformation(o + u))
            return a_hat
        
        return o