# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 17:21:36 2021

@author: Gabri
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import umap
import umap.plot

import config as cfg


class DataLoader:

    def __init__(self):

        self.train_reads, self.test_reads = self.get_reads()

        self.kmeans = {}

        self.train_labels = pd.read_csv(cfg.TRAIN_FILES["labels"], header=None).to_numpy()
        self.train_labels = self.train_labels.reshape((self.train_labels.shape[0])) - 1

        self.test_labels = pd.read_csv(cfg.TEST_FILES["labels"], header=None).to_numpy()
        self.test_labels = self.test_labels.reshape((self.test_labels.shape[0])) - 1

    def get_labels(self):
        return self.train_labels, self.test_labels

    def get_reads(self):
        train_reads = {}
        test_reads = {}

        for inp, val in cfg.INPUT_SELECTION.items():
            if val:

                if inp == "total_acc":
                    train_reads["total_acc"] = [pd.read_csv(cfg.TRAIN_FILES["total_acc_x"], header=None).to_numpy(),
                                                pd.read_csv(cfg.TRAIN_FILES["total_acc_y"], header=None).to_numpy(),
                                                pd.read_csv(cfg.TRAIN_FILES["total_acc_z"], header=None).to_numpy()]

                    test_reads["total_acc"] = [pd.read_csv(cfg.TEST_FILES["total_acc_x"], header=None).to_numpy(),
                                               pd.read_csv(cfg.TEST_FILES["total_acc_y"], header=None).to_numpy(),
                                               pd.read_csv(cfg.TEST_FILES["total_acc_z"], header=None).to_numpy()]
                elif inp == "body_acc":
                    train_reads["body_acc"] = [pd.read_csv(cfg.TRAIN_FILES["body_acc_x"], header=None).to_numpy(),
                                               pd.read_csv(cfg.TRAIN_FILES["body_acc_y"], header=None).to_numpy(),
                                               pd.read_csv(cfg.TRAIN_FILES["body_acc_z"], header=None).to_numpy()]

                    test_reads["body_acc"] = [pd.read_csv(cfg.TEST_FILES["body_acc_x"], header=None).to_numpy(),
                                              pd.read_csv(cfg.TEST_FILES["body_acc_y"], header=None).to_numpy(),
                                              pd.read_csv(cfg.TEST_FILES["body_acc_z"], header=None).to_numpy()]
                elif inp == "gyro":
                    train_reads["gyro"] = [pd.read_csv(cfg.TRAIN_FILES["gyro_x"], header=None).to_numpy(),
                                           pd.read_csv(cfg.TRAIN_FILES["gyro_y"], header=None).to_numpy(),
                                           pd.read_csv(cfg.TRAIN_FILES["gyro_z"], header=None).to_numpy()]

                    test_reads["gyro"] = [pd.read_csv(cfg.TEST_FILES["gyro_x"], header=None).to_numpy(),
                                          pd.read_csv(cfg.TEST_FILES["gyro_y"], header=None).to_numpy(),
                                          pd.read_csv(cfg.TEST_FILES["gyro_z"], header=None).to_numpy()]

        return train_reads, test_reads

    def get_data(self):
        train_obs = []
        test_obs = []

        for key, tables in self.train_reads.items():
            train_obs += self.get_observations_tables(tables[0], tables[1], tables[2], key)

        for key, tables in self.test_reads.items():
            test_obs += self.get_observations_tables(tables[0], tables[1], tables[2], key, train=False)

        train_obs = np.array(train_obs)
        test_obs = np.array(test_obs)

        if cfg.PLOT_INPUT:
            self.plot_data(train_obs)
            self.plot_data(test_obs)

        return train_obs, test_obs

    def get_observations_tables(self, x_reads, y_reads, z_reads, d_type, train=True):

        if cfg.RAW_DATA:

            new_shape = (-1, cfg.N_READS, 1)

            if cfg.GAUSSIAN_HMM:

                x_table = x_reads.reshape(new_shape)
                y_table = y_reads.reshape(new_shape)
                z_table = z_reads.reshape(new_shape)

                obs = [x_table, y_table, z_table]

                if cfg.MODULE:
                    module_reads = self.module(x_reads, y_reads, z_reads)
                    obs.append(module_reads.reshape(new_shape))

            else:

                x_obs_list = np.reshape(x_reads, (-1, 1))
                y_obs_list = np.reshape(y_reads, (-1, 1))
                z_obs_list = np.reshape(z_reads, (-1, 1))

                print("Computing clusters...", end=" ")
                x_obs_list = self.get_clusters(x_obs_list, key=d_type + "_x", train=train)
                y_obs_list = self.get_clusters(y_obs_list, key=d_type + "_y", train=train)
                z_obs_list = self.get_clusters(z_obs_list, key=d_type + "_z", train=train)

                if cfg.MODULE:
                    module_reads = self.module(x_reads, y_reads, z_reads)
                    module_obs_list = np.reshape(module_reads, (-1, 1))
                    module_obs_list = self.get_clusters(module_obs_list, key=d_type + "_m", train=train)

                print("Done")

                x_table = x_obs_list.reshape(new_shape)
                y_table = y_obs_list.reshape(new_shape)
                z_table = z_obs_list.reshape(new_shape)

                obs = [x_table, y_table, z_table]

                if cfg.MODULE:
                    module_table = module_obs_list.reshape(new_shape)
                    obs.append(module_table)

        else:

            x_obs_list = self.compute_features(x_reads)
            y_obs_list = self.compute_features(y_reads)
            z_obs_list = self.compute_features(z_reads)

            if cfg.MODULE:
                module_reads = self.module(x_reads, y_reads, z_reads)
                module_obs_list = self.compute_features(module_reads)

            new_shape = (-1, cfg.N_SUBSEQUENCES, x_obs_list.shape[1])

            if not cfg.GAUSSIAN_HMM:

                x_obs_list = self.get_clusters(x_obs_list, key=d_type + "_x", train=train)
                y_obs_list = self.get_clusters(y_obs_list, key=d_type + "_y", train=train)
                z_obs_list = self.get_clusters(z_obs_list, key=d_type + "_z", train=train)

                if cfg.MODULE:
                    module_obs_list = self.get_clusters(module_obs_list, key=d_type + "_m", train=train)

                new_shape = (-1, cfg.N_SUBSEQUENCES, 1)

            x_table = x_obs_list.reshape(new_shape)
            y_table = y_obs_list.reshape(new_shape)
            z_table = z_obs_list.reshape(new_shape)

            obs = [x_table, y_table, z_table]

            if cfg.MODULE:
                module_table = module_obs_list.reshape(new_shape)
                obs.append(module_table)

        return obs

    def plot_data(self, obs, train=True):
        data = np.swapaxes(obs, 0, 1)
        data = np.reshape(data, (data.shape[0], -1))
        mapper = umap.UMAP().fit(data)
        umap.plot.points(mapper, labels=self.train_labels if train else self.test_labels)

    def module(self, dim1, dim2, dim3):
        return np.sqrt(np.square(dim1) + np.square(dim2) + np.square(dim3))

    def compute_features(self, table):
        obs = []

        for row in table:
            for subsequence in np.reshape(row, (-1, cfg.SUBSEQUENCE_LENGTH)):

                feat_list = []

                for feature, value in cfg.FEATURES.items():
                    if value:

                        if feature == "mean":
                            feat_list.append(np.mean(subsequence))
                        elif feature == "std":
                            feat_list.append(np.std(subsequence))
                        elif feature == "min":
                            feat_list.append(min(subsequence))
                        elif feature == "max":
                            feat_list.append(max(subsequence))

                obs.append(np.array(feat_list))

        return np.array(obs)

    def get_clusters(self, obs_list, key, train=True):

        X = np.array(obs_list)

        if train:
            self.kmeans[key] = KMeans(n_clusters=cfg.K, random_state=0).fit(X)

        return self.kmeans[key].predict(X)

