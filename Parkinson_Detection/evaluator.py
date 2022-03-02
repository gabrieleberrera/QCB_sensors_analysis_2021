# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:39:36 2021

@author: gabri
"""


from sklearn import metrics
import pandas as pd
import numpy as np


class Evaluator():
    
    
    def __init__(self):       
        
        self.y_true = []
        self.y_pred = []
        
        self.results = {}        

     
    
    def compute_metrics(self, y_true, y_pred):
        
        self.y_true = y_true
        self.y_pred = y_pred
       
        self.results["precision"] = metrics.precision_score(y_true, y_pred)
        self.results["recall"] = metrics.recall_score(y_true, y_pred)
        self.results["accuracy"] = metrics.accuracy_score(y_true, y_pred)
        self.results["auc_roc"] = metrics.roc_auc_score(y_true, y_pred)
        self.results["auc_pr"] = metrics.average_precision_score(y_true, y_pred)
        
        
    
    def print_evaluation(self):
        
        print("EVALUATION RESULTS:")
        print("\n Precision: \t{:.2f}".format(self.results["precision"]))
        print(" Recall: \t\t{:.2f}".format(self.results["recall"]))
        print(" Accuracy: \t\t{:.2f}".format(self.results["accuracy"]))
        print("\n AUC(ROC): \t\t{:.2f}".format(self.results["auc_roc"]))
        print(" AUC(PR): \t\t{:.2f}".format(self.results["auc_pr"]))
        
        
    def save_evaluation(self, file):
        with open(file, "w") as f:
            f.write("EVALUATION RESULTS:\n")
            f.write("\n Precision: \t{:.2f}\n".format(self.results["precision"]))
            f.write(" Recall: \t\t{:.2f}\n".format(self.results["recall"]))
            f.write(" Accuracy: \t\t{:.2f}\n".format(self.results["accuracy"]))
            f.write("\n AUC(ROC): \t\t{:.2f}\n".format(self.results["auc_roc"]))
            f.write(" AUC(PR): \t\t{:.2f}\n".format(self.results["auc_pr"]))
   
    def get_evaluation_dict(self):
        return self.results
        
    
    
def average_evaluations(evaluations, mode = "print", file = ""):
    
    n = len(evaluations)
    
    avg_metrics_dict = {
        "avg_precision": 0.0,
        "avg_recall": 0.0,
        "avg_accuracy": 0.0,
        "avg_auc_roc": 0.0,
        "avg_auc_pr": 0.0
    }
    
    
    for evaluation in evaluations:
        
        metrics_dict = evaluation.get_evaluation_dict()
        
        avg_metrics_dict["avg_precision"] += metrics_dict["precision"]
        avg_metrics_dict["avg_recall"] += metrics_dict["recall"]
        avg_metrics_dict["avg_accuracy"] += metrics_dict["accuracy"]
        avg_metrics_dict["avg_auc_roc"] += metrics_dict["auc_roc"]
        avg_metrics_dict["avg_auc_pr"] += metrics_dict["auc_pr"]
       
    avg_metrics_dict = {k: v / n for k, v in avg_metrics_dict.items()}
    
    
    if file != "":
        
        with open(file, "w") as f:
            
            f.write("EVALUATION RESULTS:\n")
            f.write("\n Avg. Precision: \t{:.2f}\n".format(avg_metrics_dict["avg_precision"]))
            f.write(" Avg. Recall: \t\t{:.2f}\n".format(avg_metrics_dict["avg_recall"]))
            f.write(" Avg. Accuracy: \t{:.2f}\n".format(avg_metrics_dict["avg_accuracy"]))
            f.write("\n Avg. AUC(ROC): \t{:.2f}\n".format(avg_metrics_dict["avg_auc_roc"]))
            f.write(" Avg. AUC(PR): \t\t{:.2f}\n".format(avg_metrics_dict["avg_auc_pr"]))
    
    
    if mode == "print":
        
        print("EVALUATION RESULTS:")
        print("\n Avg. Precision: \t{:.2f}".format(avg_metrics_dict["avg_precision"]))
        print(" Avg. Recall: \t\t{:.2f}".format(avg_metrics_dict["avg_recall"]))
        print(" Avg. Accuracy: \t{:.2f}".format(avg_metrics_dict["avg_accuracy"]))
        print("\n Avg. AUC(ROC): \t{:.2f}".format(avg_metrics_dict["avg_auc_roc"]))
        print(" Avg. AUC(PR): \t\t{:.2f}".format(avg_metrics_dict["avg_auc_pr"]))
        
    elif mode == "dict":
        
        return avg_metrics_dict
    
    elif mode == "df":
        
        return pd.DataFrame(avg_metrics_dict)
    
    
            
    
    
def evaluate(predictions_file, labels_file):
    
    print("\n- EVALUATING -\n")
    
    # Load predictions
    predictions = np.load(predictions_file)
    
    # Load labels    
    test_labels = np.load(labels_file)   
    
    evaluator = Evaluator()
    evaluator.compute_metrics(test_labels, predictions)
    
    return evaluator
    
    
    
    
    