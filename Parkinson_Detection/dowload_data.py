# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 17:58:44 2021

@author: gabri
"""

import synapseclient
import pandas as pd


# Set path
json_path = "dataset/mPower/jsons/"

# Set Synapse username and password
syn = synapseclient.login("user", "pass")

series_cols = ["deviceMotion_walking_outbound.json.items",
                  "deviceMotion_walking_return.json.items",
                  "deviceMotion_walking_rest.json.items"]

w_table = syn.tableQuery('SELECT * FROM syn5511449')

df = w_table.asDataFrame()

file_map = syn.downloadTableColumns(w_table, series_cols, json_path)

# tapResults = {handle: json.load(open(f)) for handle, f in tapMap.items()}

# csv_path = "dataset/mPowerData/files_map.csv"
# original_map = pd.read_csv(csv_path)

map_data = []

for id, path in file_map.items():
    print("\n", id)
    map_data.append([id,path])

map_df = pd.DataFrame(map_data, columns=["id", "file_path"])

map_df.to_csv("dataset/mPowerData/files_map.csv", index=False)

















