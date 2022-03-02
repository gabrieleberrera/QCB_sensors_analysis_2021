# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:16:08 2021

@author: gabri
"""

import pandas as pd
import numpy as np
import csv


# Read a single table.
# @seq_length determines the selected fixed length of each sequence 
# (shorter sequences are filled with zeros to reach @seq_length)
def get_table(file_path, subset_inds = None, seq_length = 4000):
    
    with open(file_path, 'r') as csvfile:
        
        table = []
        
        reader = csv.reader(csvfile, delimiter=',')
        
        table = []
        
        i = 0
        
        for row in reader:
            
            if not subset_inds or i in subset_inds:
                
                row = list(map(float, row))
                row = row[::2]                              # Downsample from 100Hz to 50Hz
                row = (row - np.mean(row)) / np.std(row)    # Normalization
                
                seq = np.zeros(seq_length)
                l = len(row)
                
                if l >= seq_length:               
                    seq[:] = row[:seq_length].astype(float)
                    
                else:
                    seq[:l] = row
                    
                table.append(seq)
                
            i += 1
            
        return np.array(table)
                
            
            
# Get the full dataset.
# It can be of @seq_type euqal to "out" (for outbound), "rtn" (for return) and "rest" (for resting).
def get_dataset(path, seq_type, subset_inds = None):
    
    print("Extracting {} sequences:".format(seq_type))
    
    # Get all the acceleration and rotation rate tabels 
    print("  Getting x-axis acceleration sequences...", end = " ")
    table_acc_x = get_table(path + seq_type + "_acc_x.csv", subset_inds)
    print("Done")
    
    print("  Getting y-axis acceleration sequences...", end = " ")
    table_acc_y = get_table(path + seq_type + "_acc_y.csv", subset_inds)
    print("Done")
    
    print("  Getting z-axis acceleration sequences...", end = " ")
    table_acc_z = get_table(path + seq_type + "_acc_z.csv", subset_inds)
    print("Done")
    
    print("  Getting x-axis rotation rate sequences...", end = " ")
    table_gyro_x = get_table(path + seq_type + "_gyro_x.csv", subset_inds)
    print("Done")
    
    print("  Getting y-axis rotation rate sequences...", end = " ")
    table_gyro_y = get_table(path + seq_type + "_gyro_y.csv", subset_inds)
    print("Done")
    
    print("  Getting z-axis rotation rate sequences...", end = " ")
    table_gyro_z = get_table(path + seq_type + "_gyro_z.csv", subset_inds)
    print("Done")
    
    dataset = np.zeros((table_acc_x.shape[0], table_acc_x.shape[1], 6))
    
    dataset[:,:,0] = table_acc_x
    dataset[:,:,1] = table_acc_y
    dataset[:,:,2] = table_acc_z
    dataset[:,:,3] = table_gyro_x
    dataset[:,:,4] = table_gyro_y
    dataset[:,:,5] = table_gyro_z
    
    return dataset.astype(np.float32)


# Returns a 1-D array containing the labels
def read_array(file_path):
    
    arr = pd.read_csv(file_path, header = None)
    
    return arr.values.flatten()


# Return a random array of N indeces to select a subset of the data
def select_subset(labels, N = 5000):
    
    # Select euqlal amount of True and Flase labels
    true_inds = np.argwhere(labels == True)
    false_inds = np.argwhere(labels == False)
    
    np.random.shuffle(true_inds)
    np.random.shuffle(false_inds)
    
    n = N//2
    
    inds = np.concatenate((true_inds[:n], false_inds[:n])).flatten()
    np.random.shuffle(inds)
    
    return inds
  

# Split data in train and test partitions
def split_train_test(test_partition = 0.5):
    
    labels = np.load("data/labels.npy")
    n_samples = labels.shape[0]
    
    inds = np.arange(n_samples)
    np.random.shuffle(inds)
    
    split = int(n_samples * test_partition)
    
    train_inds = inds[:split]
    test_inds = inds[split:]
    
    np.save("data/train_inds", train_inds)
    np.save("data/test_inds", test_inds)
    
    return train_inds, test_inds


# Split data in train and test partitions depending on the subject (partecipant)
def split_train_test_by_subject(test_partition = 0.5):
    
    codes = np.load("data/codes.npy", allow_pickle=True)
    
    unique_codes = np.unique(codes) 
    n_codes = unique_codes.shape[0]
    
    np.random.shuffle(unique_codes)    
    split = int(n_codes * test_partition)
    
    train_codes = unique_codes[:split]
    
    train_inds = []
    test_inds = []
    
    for i, code in enumerate(codes):
        
        if code in train_codes:
            train_inds.append(i)
        else:
            test_inds.append(i)
    
    train_inds = np.array(train_inds)
    test_inds = np.array(test_inds)
    
    np.save("data/train_inds", train_inds)
    np.save("data/test_inds", test_inds)
    
    return train_inds, test_inds
    

def pull_predictions_by_ind(predictions, labels = None, inds = None, method = "mean"):
    
    codes = load_codes(inds)
    df = pd.DataFrame(predictions)
    df["code"] = codes
    
    if labels is not None:
        df["label"] = labels
    
    if method == "mean":
        grouped_data = df.groupby("code").mean().to_numpy().astype('float32')
    elif method == "max":
        grouped_data = df.groupby("code").max().to_numpy().astype('float32')
      
    if labels is not None:
        new_predictions = grouped_data[:,:-1]
        new_labels = grouped_data[:,-1]
        
        return new_predictions, new_labels
    else:
        return grouped_data
    
    
def pull_predictions_repetead(predictions, inds = None, method = "mean"):
    
    codes = load_codes(inds)
    df = pd.DataFrame(predictions)
    df["code"] = codes
    
    
    if method == "mean":
        grouped_data = df.groupby("code").mean()
    elif method == "max":
        grouped_data = df.groupby("code").max()
    
        
    res = np.array([grouped_data.loc[code].to_numpy().astype('float32') for code in codes])
    
    return res


def create_fromcsv_files(path):
    
    # Get and save labels
    labels = read_array(path + "labels.csv") 
    np.save("data/labels", labels) 

    # Get and save codes
    codes = read_array(path + "codes.csv") 
    np.save("data/codes", codes)      
    
    # Save outbound sequences
    data = get_dataset(path + "outbound/", "out")    
    np.save("data/out_data", data)
       
    # Save rteurn sequences
    data = get_dataset(path + "return/", "rtn")    
    np.save("data/rtn_data", data)
       
    # Save rest sequences
    data = get_dataset(path + "rest/", "rest")    
    np.save("data/rest_data", data)

 

def get_train_inds():
    
    return np.load("data/train_inds.npy")



def get_test_inds():
    
    return np.load("data/test_inds.npy")



def load_labels(subset = None):
    
    labels = np.load("data/labels.npy")
    
    if subset is not None:
        labels = labels[subset]
        
    return labels


def load_codes(subset = None):
    
    codes = np.load("data/codes.npy")
    
    if subset is not None:
        codes = codes[subset]
        
    return codes



def load_data(data_type, subset = None, signal = "all"):
    
    data = np.load("data/" + data_type + "_data.npy")
    
    if subset is not None: 
        data = data[subset]
        
    if signal == "acc":
        
        return data[:,:,:3]
    
    elif signal == "rot":
        
        return data[:,:,3:]
    
    elif signal == "all":
        
        return data
    
    else:
        
        raise Exception("Argument Error: signals can be 'all', 'acc' or 'rot'")



# import config as cfg
# create_fromcsv_files(cfg.CSV_PATH)

if __name__ == "__main__":
    
    codes = ['a','a','a','b','b','c','c']
    
    predictions = np.random.randint(1,10,size=(7,3))
    
    new = pull_predictions_repetead(predictions, codes, method="max")
    
    print(predictions, "\n\n")
    print(new)
    