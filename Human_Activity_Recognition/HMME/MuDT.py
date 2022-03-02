# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 15:39:54 2021

@author: Gabri
"""

from DecisionProfile import DecisionProfile
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv

class MuDT():
    
    def __init__(self, DP_list, K, L, C, file = None):
        
        self.L = L
        self.C = C
        self.K = K
        self.file = file
        
        self.mu_DT = self.__create_mu_DT(DP_list)
        
    
    def __create_mu_DT(self,DP_list):
        
        kmeans = KMeans(n_clusters=self.K, random_state=0).fit(DP_list)
        
        if self.file:
            with open(self.file, 'a', newline ='') as f: 
                write = csv.writer(f) 
                write.writerow(kmeans.labels_) 
        
        return kmeans.cluster_centers_
    
    
    def distance(self, i, DP):
        
        dis = np.square(self.mu_DT[i] - DP.get_DP())
        dis = np.sum(dis)
        dis = np.sqrt(dis)
        
        return dis
    
    
    def all_distances(self, DP):
        
        res = []
        
        for i in range(len(self.mu_DT)):
            res.append(self.distance(i,DP))
        
        return np.array(res)
    
    
    def get_mu_DT(self):
        
        return self.mu_DT
    
    
    def print_DT(self, index):
        
        if index >= 0 and index < len(self.mu_DT):
            
            a = self.mu_DT[index].reshape((self.L, self.C))
            cols = ["C"+str(x) for x in range(self.C)]
            inds = ["L"+str(x) for x in range(self.L)]
            DP_df = pd.DataFrame(a, columns = cols, index = inds)
            
            print(DP_df)
            
        else:
            
            print("Error: index out of bound")
    
    
    