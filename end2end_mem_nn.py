#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 22:43:23 2019

@author: assiene
"""

from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.torch import padded_3d
from parlai.core.logs import TensorboardLogger

from parlai.utils.torch import (
    argsort,
    padded_tensor,
)


import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from StackedMemoryHop import StackedMemoryHop

from tensorboardX import SummaryWriter

losses = []

class End2endMemNnAgent(TorchRankerAgent):
    
    @staticmethod    
    def add_cmdline_args(argparser):
        TorchRankerAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group("End2End MemNN Arguments")
        
        agent.add_argument("-wt", "--weight-tying", type=str, default="layer-wise", help="Type of weight tying")
        agent.add_argument("-nmh", "--num-memory-hops", type=int, default=3, help="Number of memory hops")
        agent.add_argument("-tmpe", "--temporal-encoding", type=str, default="yes", help="Use temporal encoding")
        agent.add_argument("-edim", "--embedding-dim", type=int, default=100, help="Size of the embedding")
        agent.add_argument("-mtt", "--mhlayers-to-tensorboard", type=bool, default=False, help="Save memory hops layers to tensorboard")

        argparser.set_defaults(
            split_lines=True, add_p1_after_newln=True, encode_candidate_vecs=True
        )
        
        End2endMemNnAgent.dictionary_class().add_cmdline_args(argparser)
        
        return agent
    
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        
        if opt['tensorboard_log'] is True:
            self.writer = TensorboardLogger(opt)
            
         # default one does not average
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)
        torch.manual_seed(123)

    def share(self):
        shared = super().share()
        shared['model'] = self.model
        shared['writer'] = self.writer
        return shared    

    def build_model(self):
        """This function is required to build the model and assign to the
        object `self.model`.
        """

        self.learning_rate = self.opt["learningrate"]
        self.batch_size = self.opt["batchsize"]

        self.dictionnary_size = len(self.dict)
        self.embedding_dim = self.opt["embedding_dim"]
        self.K = self.opt["num_memory_hops"]
        self.weight_tying = self.opt["weight_tying"]
        self.use_temporal_encoding = True if self.opt["temporal_encoding"] == "yes" else False
        self.save_mhlayers_to_tensorboard = self.opt["mhlayers_to_tensorboard"]
        
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
        
        self.stacked_memory_hop = StackedMemoryHop(self.K, self.dictionnary_size, self.embedding_dim, self.weight_tying, use_temporal_encoding=self.use_temporal_encoding)
        self.stacked_memory_hop.apply(weight_init)

        self.model = self.stacked_memory_hop
        self.model.share_memory()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.writer = SummaryWriter()
        
        return self.model
        
        
    def vectorize(self, *args, **kwargs):
        """Override options in vectorize from parent."""
        kwargs['add_start'] = False
        kwargs['add_end'] = False
        return super().vectorize(*args, **kwargs)

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """This function takes in a Batch object as well as a Tensor of
        candidate vectors. It must return a list of scores corresponding to
        the likelihood that the candidate vector at that index is the
        proper response. If `cand_encs` is not None (when we cache the
        encoding of the candidate vectors), you may use these instead of
        calling self.model on `cand_vecs`.
        """

        questions = batch.text_vec
        contexts = padded_3d(batch.memory_vecs)

        self.position_encoding_matrix = torch.zeros([self.embedding_dim, contexts.shape[2]])
        for k in range(1, self.position_encoding_matrix.shape[0] + 1):
            for j in range(1, self.position_encoding_matrix.shape[1] + 1):
                jJ = j / self.position_encoding_matrix.shape[1]
                self.position_encoding_matrix[k-1,j-1] = (1 - jJ) - (k/self.position_encoding_matrix.shape[0]) * (1 - 2 * jJ)
        
        if self.use_cuda:
            contexts = contexts.cuda()
            self.position_encoding_matrix = self.position_encoding_matrix.cuda()

        self.model.memory_hop_layers[0].position_encoding_matrix = self.position_encoding_matrix.t()

        output = self.model(questions, contexts)
        pred = output.argmax(dim=2)

        # if self.batch_index % 100 == 0:
        #     torch.save(self.model.state_dict(), f"/scratch/jassiene/MacNetTextExperiments/model_{self.batch_index}.pth")
        # self.batch_index+= 1
        
        return output.squeeze()

    def train_step(self, batch):
        
        out = super().train_step(batch)
        
        if self.save_mhlayers_to_tensorboard:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.batch_iter)
                #self.writer.add_histogram(name + "_grad", param.grad.clone().cpu().data.numpy(), self.batch_iter)
                for memory_hop_layer in self.model.memory_hop_layers:
                    for name_in, param_in in memory_hop_layer.named_parameters():
                        self.writer.add_histogram(name_in, param_in.clone().cpu().data.numpy(), self.batch_iter)
                        #self.writer.add_histogram(name_in + "_grad", param_in.grad.clone().cpu().data.numpy(), self.batch_iter)
        
        
        return out
        
    
    def batchify(self, obs_batch, sort=False):
        """
        Override so that we can add memories to the Batch object.
        """
        batch = super().batchify(obs_batch, sort)

        # get valid observations
        valid_obs = [(i, ex) for i, ex in enumerate(obs_batch) if self.is_valid(ex)]

        if len(valid_obs) == 0:
            return batch

        valid_inds, exs = zip(*valid_obs)

        # get memories for the valid observations
        mems = None
        if any('memory_vecs' in ex for ex in exs):
            mems = [ex.get('memory_vecs', None) for ex in exs]
        batch.memory_vecs = mems
        return batch

    def _set_text_vec(self, obs, history, truncate):
        """
        Override from Torch Agent so that we can use memories.
        """
        if 'text' not in obs:
            return obs

        if 'text_vec' not in obs:
            # text vec is not precomputed, so we set it using the history
            obs['full_text'] = history.get_history_str()
            history_vecs = history.get_history_vec_list()
            if len(history_vecs) > 0:
                obs['memory_vecs'] = history_vecs[:-1]
                obs['text_vec'] = history_vecs[-1]
            else:
                obs['memory_vecs'] = []
                obs['text_vec'] = []

        # check truncation
        if 'text_vec' in obs:
            truncated_vec = self._check_truncate(obs['text_vec'], truncate, True)
            obs.force_set('text_vec', torch.LongTensor(truncated_vec))

        if 'memory_vecs' in obs:
            obs.force_set(
                'memory_vecs',
                [
                    torch.LongTensor(self._check_truncate(m, truncate, True))
                    for m in obs['memory_vecs']
                ],
            )

        return obs

    