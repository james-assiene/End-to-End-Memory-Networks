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
        self.position_encoding_matrix = None
        self.memory_size = 320
        self.temporal_encoding_matrix_memory = torch.zeros([self.memory_size, embedding_size])
        self.temporal_encoding_matrix_output = torch.zeros([self.memory_size, embedding_size])
        
        nn.init.xavier_normal_(self.temporal_encoding_matrix_memory)
        nn.init.xavier_normal_(self.temporal_encoding_matrix_output)
        
        self.temporal_encoding_memory = nn.Parameter(self.temporal_encoding_matrix_memory)
        self.temporal_encoding_output = nn.Parameter(self.temporal_encoding_matrix_output)
        
        if is_final_layer:
            self.final_affine_transformation = nn.Linear(embedding_size, dictionnary_size)
            
        
    def forward(self, x, u):
        m = self.memory_embedding(x)
        
        if self.position_encoding_matrix is not None:
            m = m * self.position_encoding_matrix
            
        m = m.sum(dim=2)
        #m+=  + self.temporal_encoding_memory[:m.shape[1]]
        c = self.output_embedding(x).sum(dim=2)
        #c+=  + self.temporal_encoding_output[:m.shape[1]]
        
        p = F.softmax(u.bmm(m.transpose(1,2)), dim=2)
        
        o = (p.transpose(1,2) * c).sum(dim=1).unsqueeze(dim=1)
        
        if self.is_final_layer:
            a_hat = F.log_softmax(self.final_affine_transformation(o + u), dim=2)
            return a_hat
        
        return o