# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 14:21:24 2021

@author: Gabri
"""

from DecisionProfile import DecisionProfile
import numpy as np

class DecisionTemplate():
    
    def __init__(self, DP_list, L, C, label):
        self.label = label
        self.L = L
        self.C = C
        self.DT = np.zeros((L,C))
        self.create_DT(DP_list)
        
    def create_DT(self, DP_list):
        for dp in DP_list:
            self.DT = self.DT + dp.get_DP()
        self.DT /= len(DP_list)
        
    def distance(self, DP):
        diff = np.square(self.DT - DP.get_DP())
        diff = np.sum(diff)
        d = np.sqrt(diff)
        
        return d
    
    def print_DT(self):
        print(self.DT)
    
        