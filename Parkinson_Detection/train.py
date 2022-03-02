# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 17:03:10 2021

@author: gabri
"""
 
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
import pickle

import model as net
import config as cfg
import data_manager as dm

import math

import random



def norm_axis(a,b,c):
    newa=a/(math.sqrt(float(a*a+b*b+c*c)))
    newb=b/(math.sqrt(float(a*a+b*b+c*c)))
    newc=c/(math.sqrt(float(a*a+b*b+c*c)))
    
    return ([newa,newb,newc])


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)], 
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)], 
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def rand_rotate(image): ## theta: angle, a, b, c, eular vector
    theta = random.random()*math.pi*2
    a = random.random()
    b = random.random()
    c = random.random()
    
    axis=norm_axis(a,b,c)
    rot_matrix = rotation_matrix(axis, theta).T
    
    if image.shape[1] == 6:
        
        # Multipy the rotation matrix for both acceleration and rotation signals
        imagenew_acc = np.dot(image[:,:3], rot_matrix)
        imagenew_rot = np.dot(image[:,3:],rot_matrix)
        
        return np.concatenate([imagenew_acc, imagenew_rot], 1)
    
    elif image.shape[1] == 3:
        
        return np.dot(image, rot_matrix)
    
    else:
        
        # Implement for other types of data with different channels
        
        raise Exception("Cahnnels Error: rotateC works only on sequences with 3 or 6 channels")


def rand_rescale_frequency(image,min_scale, max_scale):
    r = random.random()
    scale = r * (max_scale - min_scale) + min_scale
    
    [x,y]= image.shape
    y1=y
    x1=int(x*scale)
    image=cv2.resize(image,(y1,x1))
    new=np.zeros((x,y))
    if (x1>x):
        start=0
        end=start+x
        new=image[start:end]
    else:
        new_start=0
        new_end=new_start+x1
        new[new_start:new_end]=image
    return new



# Random sequence rescaling by magnitude
def rand_rescale_magnitude(signals, min_scale, max_scale):
    
    r = random.random()
    scale = r * (max_scale - min_scale) + min_scale
    
    new_signals = signals * scale 
    
    return new_signals


# Ramdom split indeces
def split_inds(n_samples, val_split = 0.5):
    
    inds = np.arange(n_samples)

    np.random.shuffle(inds)
    
    split = int(n_samples * val_split)
    
    return inds[:split], inds[split:]



# Random split data in train and validation sets
def split_train_val(dataset, n_samples, val_split = 0.5):  

    train_size = int(n_samples * 0.5)
    
    train_set = dataset.shuffle(n_samples)
    
    val_set = train_set.skip(train_size)
    train_set.take(train_size)
    
    return train_set, val_set




 # Split data in train and validation sets depending on fold number for K-fold Cross-validation
def split_train_val_CV(dataset, fold, n_folds, fold_size):  

    if fold == 0:
        
        train_set = dataset.skip(fold_size)
        val_set = dataset.take(fold_size)
            
    elif fold == (n_folds - 1):
        
        train_set = dataset.take(fold * fold_size)
        val_set = dataset.skip(fold * fold_size)
        
    else:
        
        train_set = dataset.take(fold * fold_size) \
                    .concatenate(dataset.skip(fold * fold_size + fold_size))
        val_set = dataset.skip(fold * fold_size).take(fold_size)
        
    return train_set, val_set




def run(fold):   
    
    print("\n- TRAINING FOLD {} -\n".format(fold))
    
    # Load train indeces
    train_inds = dm.get_train_inds()
    
    # Load labels    
    labels = dm.load_labels(train_inds)
    n_samples = labels.shape[0]
    
    # Set training parameters    
    n_models = cfg.N_MODELS    
    epoches = cfg.EPOCHES    
    batch_size = cfg.BATCH_SIZE
    input_types = cfg.INPUT_TYPES
    signal = cfg.SIGNAL
    min_scale = cfg.MIN_SCALE
    max_scale = cfg.MAX_SCALE
    
    # arry for the predictions
    cnn_preds = np.zeros((n_samples, len(input_types) * n_models))
    
    # Run a cross-validation for each sequences type
    for type_index, data_type in enumerate(input_types):
        
        # Get data and shuffle
        data = dm.load_data(data_type, train_inds, signal)
        seq_len = data.shape[1]
        seq_channels = data.shape[2]
    
        # Start cross-validation

        for m in range(n_models):
            
            print("Training {} data: model {}".format(data_type, m + 1))
            
            # Get train and validation indeces
            train_i, val_i = split_inds(n_samples)
            
            # split in training and validation sets
            train_set = data[train_i]
            val_set = data[val_i]
            
            # Apply transformations to the training set
            train_set = np.array([rand_rescale_frequency(x, min_scale, max_scale) for x in train_set])
            train_set = np.array([rand_rotate(x) for x in train_set])
                        
            val_set = (val_set, labels[val_i])            
            
            # Create and compile model
            model = net.get_model((seq_len, seq_channels))
            
            model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
                          loss = tf.keras.losses.BinaryCrossentropy(),
                          metrics=["accuracy"])
            
            # Create TensorBoard callback
            tb_dir = "logs/tensorboard/"
            
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1)
            
            # Create checkpoint callback
            cp_file = "trained models/{}/best_model_{}.ckpt".format(data_type, m +1)
            
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = cp_file,
                                                             save_weights_only=True,
                                                             monitor='val_loss',
                                                             mode='min',
                                                             save_best_only=True)
            
            # train model
            history = model.fit(train_set,
                      labels[train_i],
                      epochs = epoches, 
                      batch_size = batch_size, 
                      validation_data = val_set,
                      callbacks=[tensorboard_callback, cp_callback],
                      verbose = 1)
            
            log_path = "logs/fold{}/model{}/".format(fold, m)
            os.makedirs(log_path, exist_ok = True)
            
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(log_path + "history.csv")
            
            # Load best weights
            model.load_weights(cp_file)
            
            # Compute predictions
            preds = model.predict(data).flatten()
            cnn_preds[:, (type_index * n_models) + m] = preds


    print("Training Random Forest") 
    
    res_path = "results/train/fold{}/".format(fold)
    os.makedirs(res_path, exist_ok = True)


    # Train Random Forest pulling predictions by record   
    rf = RandomForestClassifier(max_depth = 5, random_state = 0)
    rf.fit(cnn_preds, labels)
    
    os.makedirs('trained models/RF/', exist_ok = True)
    with open('trained models/RF/trained_rf.sav', 'wb') as rf_file:
        pickle.dump(rf, rf_file)
      
    np.savetxt(res_path + "predictions.csv", cnn_preds, fmt='%1.4f', delimiter = ",")
    np.savetxt(res_path + "labels.csv", labels, fmt='%d', delimiter = ",")
        
    
    # Train Random Forest pulling predictions by indiviual (mean)   
    rf = RandomForestClassifier(max_depth = 5, random_state = 0)
    grouped_preds= dm.pull_predictions_repetead(cnn_preds, train_inds)
    
    rf.fit(grouped_preds, labels)
    
    with open('trained models/RF/trained_rf_ind_mean.sav', 'wb') as rf_file:
        pickle.dump(rf, rf_file)
       
    np.savetxt(res_path + "predictions_ind_mean.csv", grouped_preds, fmt='%1.4f', delimiter = ",")
    # np.savetxt(res_path + "labels_ind_mean.csv", grouped_labels, fmt='%d', delimiter = ",")
 
       
    # Train Random Forest pulling predictions by indiviual (max)   
    rf = RandomForestClassifier(max_depth = 5, random_state = 0)
    grouped_preds= dm.pull_predictions_repetead(cnn_preds, train_inds, method="max")
    
    rf.fit(grouped_preds, labels)
    
    with open('trained models/RF/trained_rf_ind_max.sav', 'wb') as rf_file:
        pickle.dump(rf, rf_file)
        
    np.savetxt(res_path + "predictions_ind_max.csv", grouped_preds, fmt='%1.4f', delimiter = ",")
    # np.savetxt(res_path + "labels_ind_max.csv", grouped_labels, fmt='%d', delimiter = ",")
        
    
    print("Training completed")  
    

    
