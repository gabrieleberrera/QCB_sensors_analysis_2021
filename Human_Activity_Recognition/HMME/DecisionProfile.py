# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:43:35 2021

@author: Gabri
"""

import pandas as pd
import numpy as np

class DecisionProfile():
    
    def __init__(self, arr, L, C):
        
        self.L = L
        self.C = C
        
        if isinstance(arr, np.ndarray) and arr.shape == (L*C,):
            
            self.DP = arr.copy()
        
        else:
            
            raise Exception("Error: insert a np.ndarray with shape (L*C,)")
        
    
    def print_DP(self): 
        
        print(self.to_data_frame())
        
    
    def to_data_frame(self):
        
        a = self.DP.reshape((self.L, self.C))
        cols = ["C"+str(x) for x in range(self.C)]
        inds = ["L"+str(x) for x in range(self.L)]
        DP_df = pd.DataFrame(a, columns = cols, index = inds)
        
        return DP_df
    
    
    def get_DP(self):
        
        return self.DP
        
    
    
    
    
    