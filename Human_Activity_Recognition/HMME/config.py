# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 14:46:14 2021

@author: gabri
"""

#### Configuration File ####

TRAIN_ROOT = "../../Data/UCI HAR Dataset/train/Inertial Signals/"
TEST_ROOT = "../../Data/UCI HAR Dataset/test/Inertial Signals/"

# Dictionary containing the files names
TRAIN_FILES = {
    "total_acc_x": TRAIN_ROOT + "total_acc_x_train.csv",
    "total_acc_y": TRAIN_ROOT + "total_acc_y_train.csv",
    "total_acc_z": TRAIN_ROOT + "total_acc_z_train.csv",
    "body_acc_x": TRAIN_ROOT + "body_acc_x_train.csv",
    "body_acc_y": TRAIN_ROOT + "body_acc_y_train.csv",
    "body_acc_z": TRAIN_ROOT + "body_acc_z_train.csv",
    "gyro_x": TRAIN_ROOT + "body_gyro_x_train.csv",
    "gyro_y": TRAIN_ROOT + "body_gyro_y_train.csv",
    "gyro_z": TRAIN_ROOT + "body_gyro_z_train.csv",
    "labels": "../../Data/UCI HAR Dataset/train/y_train.txt",
    "train_DPs": "../../Data/Saved_Objects/DPs_train.csv",
    "test_DPs": "../../Data/Saved_Objects/DPs_test.csv",
    "train_labels": "../../Data/Saved_Objects/labels_train.csv",
    "test_labels": "../../Data/Saved_Objects/labels_test.csv"
    }

TEST_FILES = {
    "total_acc_x": TEST_ROOT + "total_acc_x_test.csv",
    "total_acc_y": TEST_ROOT + "total_acc_y_test.csv",
    "total_acc_z": TEST_ROOT + "total_acc_z_test.csv",
    "body_acc_x": TEST_ROOT + "body_acc_x_test.csv",
    "body_acc_y": TEST_ROOT + "body_acc_y_test.csv",
    "body_acc_z": TEST_ROOT + "body_acc_z_test.csv",
    "gyro_x": TEST_ROOT + "body_gyro_x_test.csv",
    "gyro_y": TEST_ROOT + "body_gyro_y_test.csv",
    "gyro_z": TEST_ROOT + "body_gyro_z_test.csv",
    "labels": "../../Data/UCI HAR Dataset/test/y_test.txt",
    "train_DPs": "../../Data/Saved_Objects/DPs_train.csv",
    "test_DPs": "../../Data/Saved_Objects/DPs_test.csv",
    "train_labels": "../../Data/Saved_Objects/labels_train.csv",
    "test_labels": "../../Data/Saved_Objects/labels_test.csv"
}

# Set to True the data to use for the classification
INPUT_SELECTION = {
    "total_acc": True,
    "body_acc": True,
    "gyro": True
    }

# Numer of repeats
REPEATS = 1

# Set to use directly the raw data instead of selecting subsequences
RAW_DATA = False

# Set the length and the number of subsequences (N_READS needs to be divisible
# by SUBSEQUENCE_LENGTH)
N_READS = 128
SUBSEQUENCE_LENGTH = 16
N_SUBSEQUENCES = N_READS // SUBSEQUENCE_LENGTH

# Set True if you want to compute also the module of the acceleartions and 
# gyroscope data
MODULE = True

# If True use a Gaussian HMM otherwise a Multinomial HMM
GAUSSIAN_HMM = False

# Number of states of each HMM
STATES = 10

# Allowed methods: "knearest", "SVM", "RF"
DPS_CLASSIFIER = "RF"

# K value for k-means and k-nearest algorithms
K = 64

# Set to True the features to compute for each subsequence
FEATURES = {
    "mean": True,
    "std": True,
    "min": True,
    "max": True
    }

# Set to True if you want to save the DPs and their labels on csv files
SAVE_DPS = False

# Set to True if you want to normalize the computed DPs
NORM = True

# Plot with UMAP the input data
PLOT_INPUT = False

# Plot with UMAP the DPs
PLOT_DPs = False







