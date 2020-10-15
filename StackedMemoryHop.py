#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 23:15:16 2019

@author: assiene
"""

import numpy as np
import torch
import torch.nn as nn

from SingleMemoryHop import SingleMemoryHop

class StackedMemoryHop(nn.Module):
    
    def __init__(self, K, dictionnary_size, embedding_size, weight_tying="layer-wise", use_temporal_encoding=True):
        
        super(StackedMemoryHop, self).__init__()
        self.K = K
        self.weight_tying = weight_tying
        
        self.question_embedding = nn.Embedding(dictionnary_size, embedding_size)
        
        self.memory_hop_layers = []
        if K > 1:
            for i in range(K-1):
                memory_hop_layer = SingleMemoryHop(dictionnary_size, embedding_size, use_temporal_encoding=use_temporal_encoding)
                
                if i > 0:
                    previous_memory_hop_layer = self.memory_hop_layers[i-1]
                    if self.weight_tying == "layer-wise":
                        memory_hop_layer.memory_embedding = previous_memory_hop_layer.memory_embedding
                        memory_hop_layer.temporal_encoding_memory = previous_memory_hop_layer.temporal_encoding_memory
                        
                        memory_hop_layer.output_embedding = previous_memory_hop_layer.output_embedding
                        memory_hop_layer.temporal_encoding_output = previous_memory_hop_layer.temporal_encoding_output
                    else:
                        memory_hop_layer.memory_embedding = previous_memory_hop_layer.output_embedding
                        memory_hop_layer.temporal_encoding_memory = previous_memory_hop_layer.temporal_encoding_output
                
                self.memory_hop_layers.append(memory_hop_layer)
        
        final_memory_hop_layer = SingleMemoryHop(dictionnary_size, embedding_size, is_final_layer=True, use_temporal_encoding=use_temporal_encoding)
        self.memory_hop_layers.append(final_memory_hop_layer)
        self.memory_hop_layers = nn.ModuleList(self.memory_hop_layers)
        
        if weight_tying == "layer-wise":
            self.H = nn.Linear(embedding_size, embedding_size)
            
        else:
            self.question_embedding = self.memory_hop_layers[0].memory_embedding
            final_memory_hop_layer.final_affine_transformation.weight.data = final_memory_hop_layer.output_embedding.weight.data
    
    def forward(self, q, x):
        u = self.question_embedding(q).sum(dim=1).unsqueeze(dim=1)

        for i in range(self.K):
            memory_hop_layer = self.memory_hop_layers[i]
            o = memory_hop_layer(x, u)
            
            if i < self.K-1:
                if self.weight_tying == "layer-wise":
                    u = self.H(u) + o
                
                else:
                    u = u + o
            
        return o