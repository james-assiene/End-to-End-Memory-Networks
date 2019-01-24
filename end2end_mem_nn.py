#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:43:23 2019

@author: assiene
"""

from parlai.core.torch_agent import TorchAgent, Output
from parlai.core.utils import padded_3d
from parlai.core.logs import TensorboardLogger

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from StackedMemoryHop import StackedMemoryHop

from tensorboardX import SummaryWriter

losses = []

class End2endMemNnAgent(TorchAgent):
    
    def __init__(self, opt, shared=None):
        
        super().__init__(opt, shared)
        
        if opt['tensorboard_log'] is True:
            self.writer = TensorboardLogger(opt)
        
        self.dictionnary_size = 177
        self.embedding_dim = 100
        self.K = 3
        self.weight_tying = "layer-wise"
        self.criterion = nn.NLLLoss()
        
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
        
        self.stacked_memory_hop = StackedMemoryHop(self.K, self.dictionnary_size, self.embedding_dim, self.weight_tying)
        self.stacked_memory_hop.apply(weight_init)
        self.optimizer = optim.Adam(self.stacked_memory_hop.parameters())
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 25, 0.5)
        self.batch_iter = 0
        
        
    def vectorize(self, *args, **kwargs):
        """Override options in vectorize from parent."""
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        kwargs['split_lines'] = True
        return super().vectorize(*args, **kwargs)
        
    def train_step(self, batch):
        
        #self.scheduler.step()
        
        self.stacked_memory_hop.train()
        
        questions, answers = batch.text_vec, batch.label_vec
        contexts = padded_3d(batch.memory_vecs)
        
        loss = 0
        self.optimizer.zero_grad()

        output = self.stacked_memory_hop(questions, contexts)
        pred = output.argmax(dim=2)
        
        loss = self.criterion(output.squeeze(1), answers.squeeze(1))
        losses.append(loss.item())
        self.writer.add_scalar("data/loss", loss, self.batch_iter)
        
        for name, param in self.stacked_memory_hop.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.batch_iter)
            #self.writer.add_histogram(name + "_grad", param.grad.clone().cpu().data.numpy(), self.batch_iter)
            for memory_hop_layer in self.stacked_memory_hop.memory_hop_layers:
                for name_in, param_in in memory_hop_layer.named_parameters():
                    self.writer.add_histogram(name_in, param_in.clone().cpu().data.numpy(), self.batch_iter)
                    #self.writer.add_histogram(name_in + "_grad", param_in.grad.clone().cpu().data.numpy(), self.batch_iter)
        
        #print("Loss : ", loss.item())
        loss.backward()
        self.optimizer.step()
        
        self.batch_iter+= 1
        
        return Output(self.dict.vec2txt(pred))
    
    def eval_step(self, batch):
        questions = batch.text_vec
        contexts = padded_3d(batch.memory_vecs)

        output = self.stacked_memory_hop(questions, contexts)
        pred = output.argmax(dim=2)
        
        return Output(self.dict.vec2txt(pred))
    
    
    
from parlai.scripts.train_model import TrainLoop, setup_args

if __name__ == '__main__':
    parser = setup_args()
    opt = parser.parse_args()
    opt["tensorboard_log"] = True
    opt["model_file"] = "m1"
    opt["tensorboard_tag"] = "task,batchsize"
    opt["tensorboard_metrics"] = "all"
    opt["metrics"] = "all"
    #opt["model"] = "end2end_mem_nn"
    #opt["no_cuda"] = True
    TrainLoop(opt).train()