# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:09:14 2021

@author: Gabri
"""

import pandas as pd
import numpy as np
import time
import umap
import umap.plot

from HMM_rack import HMM_rack
from MuDT import MuDT
from Waitbar import Waitbar

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from termcolor import colored

import config as cfg

class HMME():
    
    def __init__(self, L, C):
        
        self.L = L
        self.C = C
        
        self.K = cfg.K
        self.states = cfg.STATES
        
        self.trained = False
        self.mu_DT_list = []
        self.hmm_rack = HMM_rack(L, C, self.states)
        
        self.classifier = None
        
        if cfg.NORM:
            self.norm = None
        
        
    def train(self, train_set, labels):
        
        print("HMME TRAINING: {} training samples".format(train_set.shape[1]))
        print()
        
        start_time = time.time()
        
        self.__check_train_input(train_set, labels)
        
        print("Training the rack of HMMs...")
        
        if self.__train_hmm_rack(train_set, labels):            
            print(colored(" HMM rack succesfully trained!", "green"))            
        else:            
            raise Exception("Error during HMM rack training: make sure that the " +
                            "training set is complete and in the proper form")
            
        print("\nComputing training DPs...")
        
        DPs = self.__compute_all_DPs(train_set)
        
        if cfg.SAVE_DPS:
            self.__save_DPs(DPs, cfg.TRAIN_FILES["train_DPs"])
            self.__save_labels(labels, cfg.TRAIN_FILES["train_labels"])
        
        if cfg.PLOT_DPs:
            self.__plot_DPs(DPs, labels)
        
        if cfg.DPS_CLASSIFIER == "knearest":
            
            self.classifier = self.__train_knearest_with_DTs(DPs, labels)
            
        elif cfg.DPS_CLASSIFIER == "SVM":
            
            self.classifier = SVC()
            self.classifier.fit(DPs, labels)
            
        elif cfg.DPS_CLASSIFIER == "RF":
            
            self.classifier = RandomForestClassifier()
            self.classifier.fit(DPs, labels)
        
        else:
            raise Exception("Configuration error: set DPS_CLASSIFIER " +
                            "to 'knearest', 'SVM', 'RF' in confi.py file")
            
        
        t = round(time.time() - start_time)
        m = t // 60
        s = t % 60
        print("\nTraining completed! Time: {}m {}s".format(m,s))
        print()
        
        self.trained = True
    
    
    def predict(self, test_set, labels = None):
        
        if self.trained:
            
            print("HMME PREDICTING: {} testing samples".format(test_set.shape[1]))
            print()
            
            self.__check_test_input(test_set)
            
            print("\nComputing DPs...")
            
            DPs = self.__compute_all_DPs(test_set, training = False)
            
            if cfg.SAVE_DPS:
                self.__save_DPs(DPs, cfg.TEST_FILES["test_DPs"])
                if labels is not None:
                    self.__save_labels(labels, cfg.TEST_FILES["test_labels"])

            return self.classifier.predict(DPs)
        
        else:
            
            print("Error: train HMME before using predict (use method train). Return None")
            return None
    
    
    def __plot_DPs(self, DPs, labels):
        mapper = umap.UMAP().fit(DPs)
        umap.plot.points(mapper, labels = labels)
    
    
    def __get_all_mu_DTs(self):
        
        DTs = []
        
        for mu_DT in self.mu_DT_list:
            
            mu_DT_table = mu_DT.get_mu_DT()
            
            for i in range(mu_DT_table.shape[0]):
                DTs.append(mu_DT_table[i])
                
        labels = np.repeat(np.array(range(self.C)), self.K)

        return np.array(DTs), labels
    
    def __check_test_input(self, test_set):
        
        if test_set.shape[0] != self.L:
            raise Exception("Wrong testing set dimensions")
    
    
    def __compute_all_DPs(self, data_set, training = True):
        
        waitbar = Waitbar(data_set.shape[1])
        
        DPs = []
        
        for i in range(data_set.shape[1]):
            
            obs = data_set[:,i]
            DPs.append(self.hmm_rack.compute_DP(obs).get_DP())
            
            waitbar.step(i)
            
        waitbar.stop()
        
        DPs = np.array(DPs)
        
        # Normalize DPs
        if cfg.NORM:
            if training:
                self.norm = np.linalg.norm(DPs, axis = 0)
            DPs = DPs / self.norm
             
        return DPs
    
    
    def __train_knearest_with_DTs(self, DPs, labels):
        
        all_DTs = []
        
        for c in range(self.C):
           
            inds = np.where(labels == c)[0]

            mu_dt = MuDT(DPs[inds], self.K, self.L, self.C)
            all_DTs.append(mu_dt.get_mu_DT())
        
        
        all_DTs = np.concatenate(all_DTs)
        DTs_labels = np.repeat(np.array(range(self.C)), self.K)
        
        knn = KNeighborsClassifier(n_neighbors = 3)
        knn.fit(all_DTs, DTs_labels)
            
        return knn
     
    
        
    def __save_DPs(self, DPs, dps_file):
        
        col_names = ""
    
        for l in range(self.L):
            for c in range(self.C):
                col_names += "Prob_HMM_dim{}_class{},".format(l,c)
        
        col_names = col_names[:-1]
        
        f = open(dps_file,"w")
        f.write(col_names+"\n")
        f.close()
        
        DPs_df = pd.DataFrame(DPs)        
        DPs_df.to_csv(dps_file, header = False, index = False, mode = "a")
            
            
    def __save_labels(self, labels, labels_file):
        
        labels_df = pd.DataFrame(labels)
        labels_df.to_csv(labels_file, header = False, index = False)
        
    
    def __train_hmm_rack(self, train_set, labels):
        
        seq_len = train_set.shape[2]
        
        i = 0
        
        waitbar = Waitbar(self.C * self.L)
        waitbar.start()
        
        for c in range(self.C):
            
            inds = np.where(labels == c)[0]
                        
            for l in range(self.L):
            
                subset = train_set[l,inds,:]
                obs = subset.reshape((-1, train_set.shape[3]))
                self.hmm_rack.train_HMM(obs, l, c, seq_len)
                
                i += 1
                waitbar.step(i)
                
        waitbar.stop()
                
        if not cfg.GAUSSIAN_HMM:
            self.hmm_rack.fill_gaps(self.K)
        
        return self.hmm_rack.is_trained()
    
    
    def __check_train_input(self, train_set, labels):
        
        if not (train_set.ndim == 4 and 
                labels.ndim == 1 and
                train_set.shape[1] == len(labels)):
            
            raise Exception('Incorrect input dimension')
            
        else:

            values = np.unique(labels, return_counts=False)
            
            if not np.array_equal(values, np.arange(self.C)):
                
                raise Exception('Incomplete training set')
            
            
    def get_HMM_rack(self):
        return self.hmm_rack.get_rack()
    
    
    