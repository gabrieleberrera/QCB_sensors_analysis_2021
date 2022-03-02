# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 16:53:42 2021

@author: Gabri
"""

from DataLoader import DataLoader
from HMME import HMME
import config as cfg

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def split_data(x, test_size=0.3):
    i = round(len(x) * test_size)

    a = x.copy()
    np.random.shuffle(a)
    
    return a[i:], a[:i]
    

if __name__ == "__main__":
    
    DL = DataLoader()
    
    train_obs, test_obs = DL.get_data()
    train_labels, test_labels = DL.get_labels()
    
    L = train_obs.shape[0]
    C = 6

    best_acc = 0
    best_model = None
    accs = []

    for r in range(cfg.REPEATS):
        print("Experiment #{}".format(r + 1))

        train, val = split_data(np.arange(len(train_labels)))
    
        hmme = HMME(L, C)
        hmme.train(train_obs[:, train], train_labels[train])

        preds = hmme.predict(train_obs[:, val], train_labels[val])

        acc = accuracy_score(train_labels[val], preds)
        accs.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_model = hmme

    preds = best_model.predict(test_obs, test_labels)

    print("\nCLASSIFICATION REPORT")
    print(classification_report(test_labels, preds))
    

    
