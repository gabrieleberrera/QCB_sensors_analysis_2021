# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 20:49:04 2021

@author: gabri
"""

import json
import pandas as pd
import numpy as np




def get_sample_sequences(json_path):
    
    sequences = [[] for _ in range(6)]
    
    with open(json_path, "r") as f:
        json_list = json.load(f)
        
    for json_dict in json_list:
        sequences[0].append(json_dict["userAcceleration"]["x"])
        sequences[1].append(json_dict["userAcceleration"]["y"])
        sequences[2].append(json_dict["userAcceleration"]["z"])
        sequences[3].append(json_dict["rotationRate"]["x"])
        sequences[4].append(json_dict["rotationRate"]["y"])
        sequences[5].append(json_dict["rotationRate"]["z"])

    return norm_and_pad(sequences)




def norm_and_pad(sequences):
    
    new_l = 4000
    
    new_sequences = np.zeros((new_l, 6))
    
    for i, seq in enumerate(sequences):
        
        seq = seq[::2]
        l = len(seq)
        
        if l < new_l:
            
            new_seq = (seq - np.mean(seq)) / np.std(seq)
            new_sequences[:l, i] = new_seq
            
        else:
            
            new_seq = seq[:new_l]
            new_seq = (new_seq - np.mean(new_seq)) / np.std(new_seq)
            new_sequences[:,i] = new_seq
            
    return np.array(new_sequences).astype(np.float32)




if __name__ == "__main__":
    
    path = "D:/scuola/QCB/Secondo anno/Tirocinio-QCB-2021/Data/mPowerData/"
    
    w_df = pd.read_csv(path + "walking_table.csv")
    d_df = pd.read_csv(path + "demographic_table.csv")
    f_map = pd.read_csv(path + "files_map.csv")
    f_map = {row[0]: row[1] for index, row in f_map.iterrows()}
    
    w_cols = ["recordId",
              "healthCode",
              "deviceMotion_walking_outbound.json.items",
              "deviceMotion_walking_return.json.items",
              "deviceMotion_walking_rest.json.items"]
    
    w_df = w_df[w_cols]
    d_df = d_df[["healthCode", "professional-diagnosis"]]
    
    full_table = pd.merge(w_df, d_df, on = "healthCode")
    
    full_table = full_table.dropna().convert_dtypes()
    
    labels = []
    record_codes = []
    health_codes = []
    
    out_data = []
    rtn_data = []
    rest_data = []
    
    i = 1
    nrows = full_table.shape[0]
    
    for index, row in full_table.iterrows():
        
        print("Saving {} of {}".format(i, nrows))
        i += 1
    
        record_code = row[0]
        health_code = row[1]
        label = row[-1]
        
        out_path = f_map[row[2]]
        rtn_path = f_map[row[3]]
        rest_path = f_map[row[4]]
        
        try:
            
            out_seqs = get_sample_sequences(out_path)
            rtn_seqs = get_sample_sequences(rtn_path)
            rest_seqs = get_sample_sequences(rest_path)
            
            out_data.append(out_seqs)
            rtn_data.append(rtn_seqs)
            rest_data.append(rest_seqs)
            
            labels.append(label)
            record_codes.append(record_code)
            health_codes.append(health_code)
            
        except Exception:
            
            print("Skipped record: dameged or empty files")

    
    df = pd.DataFrame((record_codes, health_codes, labels, out_data, rtn_data, rest_data), 
                      columns = ['recordID','healthID','label','out','rtn','rest'])
    
    df = pd.DataFrame({'recordID': record_codes,
                       'healthID': health_codes,
                       'label': labels,
                       'out': out_data,
                       'rtn': rtn_data,
                       'rest': rest_data})

    np.save("data/labels", np.array(labels)) 
    np.save("data/codes", np.array((record_code, health_code)))      
 
    np.save("data/out_data", np.array(out_data))
    np.save("data/rtn_data", np.array(rtn_data))  
    np.save("data/rest_data", np.array(rest_data))


    
    












