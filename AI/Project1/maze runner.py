# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:43:13 2017

@author: Kiyoon Jeong
"""
from random import randint

def createmaze(dim,p):
    size = dim**2
    
    map = dict()
    for i in range(dim):
        for j in range(dim):
            map.add([i,j])
            
    occupied = size*p
    
    block = set([])
    
    while len(block) == occupied:
        occupied.add([randint(1,size*p-1),randint(1,size*p-1)])
    
    maze = map.difference(block)
    
    return maze
    
    