# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:38:52 2021

@author: gabri
"""

class Waitbar():
    
    def __init__(self, n_iter, buff_size = 50):
        self.buff_size = buff_size
        self.step_size = buff_size / n_iter
        
    def start(self):
        s = "  |{}|".format("-" * self.buff_size)        
        print(s, end = "\r")
        
    def step(self, i):
        progress = round((i + 1) * self.step_size)
        s = "\r  |{}{}|".format("█" * progress, "-" * (self.buff_size - progress))
        print(s, end = "\r")
        
    def stop(self):
        print()