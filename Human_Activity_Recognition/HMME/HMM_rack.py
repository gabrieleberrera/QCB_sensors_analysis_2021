# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 11:39:25 2021

@author: Gabri
"""

from DecisionProfile import DecisionProfile

from hmmlearn import hmm
import numpy as np
import math

import config as cfg

class HMM_rack():
    
    def __init__(self, L, C, states):
        
        self.L = L
        self.C = C
        self.states = states
        
        self.rack = np.full((L, C), None)
        
    
    def train_HMM(self, obs, l, c, seq_len):
        
        if l in range(self.L) and c in range(self.C):
            
            if obs.shape[0] % seq_len == 0:
                
                if self.rack[l,c] != None:
                    raise Exception("Warning: HMM in position L = {} and C = {} was already trained.")
                    
                #print(len(obs) / seq_len)
                lengths = np.full(len(obs) // seq_len, seq_len)
                
                if cfg.GAUSSIAN_HMM:
                    self.rack[l,c] = hmm.GaussianHMM(n_components = self.states, n_iter = 100).fit(obs, lengths)
                else:
                    self.rack[l,c] = hmm.MultinomialHMM(n_components = self.states, n_iter = 100).fit(obs, lengths)
            
            else:
                
                raise Exception("All the observation sequences must have the same length")
                
        else:
            
            raise Exception("Index out of bound")
            
    def fill_gaps(self, K):
        
        if self.is_trained():
            
            for l in range(self.L):
                for c in range(self.C):
                    
                    emissions = self.rack[l,c].emissionprob_
                    size = emissions.shape[1]
                    
                    if size < K:
                        
                        new_emissions = np.zeros((self.states, K))
                        new_emissions[:,:size] = emissions
                        self.rack[l,c].emissionprob_ = new_emissions
                        self.rack[l,c].n_features = K
            
        else:
            
            raise Exception("the HMM rack is not completely trained")
            
            
    
    def is_trained(self):
        
        return not None in self.rack
    
   
    def compute_DP(self, obs):
        
        if self.is_trained():
        
            if obs.shape[0] != self.L:
                raise Exception("obs argument has to contain a sequence of observations for each dimension")
                    
            dp_arr = np.zeros((self.L, self.C))

            for l in range(self.L):            
                for c in range(self.C): 
                    score = self.rack[l,c].score(obs[l])
                    dp_arr[l,c] = math.exp(score)
                    
            return DecisionProfile(dp_arr.reshape(self.L * self.C), self.L, self.C)
        
        else:
            
            raise Exception("HMM rack is not completely tarined")
    
    def get_rack(self):
        return self.rack
