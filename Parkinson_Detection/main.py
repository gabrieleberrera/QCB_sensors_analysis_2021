# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 12:22:09 2021

@author: gabri
"""

import config as cfg
import train
import predict
import data_manager
import evaluator
import tensorflow as tf

import logging

tf.get_logger().setLevel(logging.ERROR)



if __name__ == "__main__":
    
    # data_manager.create_fromcsv_files(cfg.CSV_PATH)
    
    n_folds = cfg.FOLDS    
    evaluations = []
    evaluations_ind_mean = []
    evaluations_ind_max = []
    
    
    for fold in range(n_folds):
        
        res_path = "results/test/fold{}/".format(fold)
        
        data_manager.split_train_test_by_subject()
        
        train.run(fold)
        predict.run(fold)
        
        eva = evaluator.evaluate(res_path + "predictions.npy", res_path + "labels.npy")
        eva_ind_mean = evaluator.evaluate(res_path + "predictions_ind_mean.npy", res_path + "labels.npy")
        eva_ind_max = evaluator.evaluate(res_path + "predictions_ind_max.npy", res_path + "labels.npy")
        
        eva.save_evaluation(res_path + "eva.txt")
        eva_ind_mean.save_evaluation(res_path + "eva_ind_mean.txt")
        eva_ind_max.save_evaluation(res_path + "eva_ind_max.txt")
        
        evaluations.append(eva)
        evaluations_ind_mean.append(eva_ind_mean)
        evaluations_ind_max.append(eva_ind_max)
        
    evaluator.average_evaluations(evaluations, file = "results/test/avg_eva.txt")
    evaluator.average_evaluations(evaluations_ind_mean, file = "results/test/avg_eva_ind_mean.txt")
    evaluator.average_evaluations(evaluations_ind_max, file = "results/test/avg_eva_ind_max.txt")
        
        
        
        
        
        