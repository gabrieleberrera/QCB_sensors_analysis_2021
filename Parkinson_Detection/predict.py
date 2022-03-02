# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:13:31 2021

@author: gabri
"""

import numpy as np
import tensorflow as tf
import pickle
import os

import model as net
import config as cfg
import data_manager as dm



def run(fold, with_labels = True):
    
    print("\n- PREDICTING -\n")
    
    # Load train indeces
    test_inds = dm.get_test_inds()
    
    n_samples = test_inds.shape[0]
    
    # Load test labels
    if with_labels:
        labels = dm.load_labels(test_inds)
    
    # Get parameters from config file
    n_models = cfg.N_MODELS
    input_types = cfg.INPUT_TYPES
    batch_size = cfg.BATCH_SIZE  
    signal = cfg.SIGNAL

    # arry for the predictions
    cnn_preds = np.zeros((n_samples, len(input_types) * n_models))
    
    # Get predictions form the CNN ensemble
    for type_index, data_type in enumerate(input_types):
        
        data = dm.load_data(data_type, test_inds, signal)
        n_samples = data.shape[0]
        seq_len = data.shape[1]
        seq_channels = data.shape[2]
        
        
        for m in range(n_models):
            
            print(" Predicting {} data: model {}".format(data_type, m + 1))
            
            # Load trained CNN model
            model = net.get_model((seq_len, seq_channels))            
            trained_net_file = "trained models/{}/best_model_{}.ckpt".format(data_type, m +1)            
            model.load_weights(trained_net_file)
            
            # Compute predictions
            preds = model.predict(data, batch_size=batch_size)
            
            if type_index == 0 and fold == 0:
                cnn_preds = np.zeros((n_samples, len(input_types) * n_models))
            
            # Save model predictions
            cnn_preds[:, (type_index * n_models) + m] = preds.flatten()
    
    
    res_path = "results/test/fold{}/".format(fold)
    os.makedirs(res_path, exist_ok = True)
    
    
    # Load the trained Random Forest by record
    with open('trained models/RF/trained_rf.sav', 'rb') as rf_file:
        rf = pickle.load(rf_file)
        
    # Compute final predictions
    final_preds = rf.predict(cnn_preds)    
    
    # Save on files
    np.save(res_path + "predictions", final_preds.flatten())
    
    if with_labels:
        np.save(res_path + "labels", labels)
    
    
    # Load the trained Random Forest by individual (mean)
    with open('trained models/RF/trained_rf_ind_mean.sav', 'rb') as rf_file:
        rf = pickle.load(rf_file)
        
    # Compute final predictions
    grouped_preds= dm.pull_predictions_repetead(cnn_preds, inds = test_inds)
    
    final_preds = rf.predict(grouped_preds)    
    
    # Save on files
    np.save(res_path + "predictions_ind_mean", final_preds.flatten())
    
                
            
    # Load the trained Random Forest by individual (max)
    with open('trained models/RF/trained_rf_ind_max.sav', 'rb') as rf_file:
        rf = pickle.load(rf_file)
        
    # Compute final predictions
    grouped_preds= dm.pull_predictions_repetead(cnn_preds, inds = test_inds, method = "max")
    
    final_preds = rf.predict(grouped_preds)   
    
    # Save on files
    np.save(res_path + "predictions_ind_max", final_preds.flatten())
                    
            
            
            
            
            
            
            
            
            
            
            
            
            