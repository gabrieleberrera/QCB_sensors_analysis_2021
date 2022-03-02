# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:17:31 2021

@author: gabri
"""

# Dataset .csv files location
CSV_PATH = "D:/scuola/QCB/Secondo anno/Tirocinio-QCB-2021/Data/mPowerData/csv_data/"


BATCH_SIZE = 64

EPOCHES = 2

SEQ_L = 4000

FOLDS = 2

N_MODELS = 2

INPUT_TYPES = ["rest"]

SIGNAL = "acc"

# Min and max for random scaling
MIN_SCALE = 0.8
MAX_SCALE = 1.2