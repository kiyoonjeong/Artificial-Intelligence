# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:44:04 2017

@author: Kiyoon Jeong
"""
import numpy as np
from random import randint



def getMine(dimx, dimy, mines):
    
    dimx = int(dimx)
    dimy = int(dimy)
    mines = int(mines)

        # Generate a binary value matrix with certain proportion of value 1
    mine = np.random.choice(1, size=(dimx, dimy))
    
    while sum(sum(mine)) != 10:
        mine[randint(0,dimx-1), randint(0,dimy-1)] = 1
        
    return mine
    
def modMine(mine):
    mine1 = np.random.choice(1, size=(len(mine), len(mine[0])))
    for i in range(len(mine)):
        for j in range(len(mine[0])):
            if mine[i,j] == 1:
                mine1[i,j] = 9
            else :
                if i == 0:
                    if j == 0:
                        mine1[i,j] = mine[i+1,j] + mine[i,j+1] + mine[i+1,j+1]
                    if j == len(mine[0])-1:
                        mine1[i,j] = mine[i+1,j] + mine[i,j-1] + mine[i+1,j-1]
                    else:
                        mine1[i,j] = mine[i,j-1] + mine[i+1, j-1] + mine[i+1,j] + mine[i+1,j+1] + mine[i,j+1]
                elif i == len(mine)-1:
                    if j == 0:
                        mine1[i,j] = mine[i-1,j] + mine[i-1,j+1] + mine[i,j+1]
                    if j == len(mine[0])-1:
                        mine1[i,j] = mine[i-1,j] + mine[i-1,j-1] + mine[i,j-1]
                    else:
                        mine1[i,j] = mine[i,j-1] + mine[i-1, j-1] + mine[i-1,j] + mine[i-1,j+1] + mine[i,j+1]
                elif j == 0:
                    mine1[i,j] = mine[i-1,j] + mine[i-1,j+1] + mine[i,j+1] + mine[i+1,j]+ mine[i+1,j+1]
                elif j == len(mine[0])-1:
                    mine1[i,j] = mine[i-1,j] + mine[i-1,j-1] + mine[i,j-1] + mine[i+1,j-1] + mine[i+1,j]
                else:
                    mine1[i,j] = mine[i-1,j-1] + mine[i-1, j] + mine[i-1, j+1] + mine[i, j-1] + mine[i, j+1] + mine[i+1, j-1] + mine[i+1, j] + mine[i+1, j+1]
    return mine1

    
def main():
    dimx = int(input("Enter mine dimension.x : "))
    dimy = int(input("Enter mine dimension.y : "))
    numberMines = int(input("Enter number of mines : "))
    a = getMine(dimx, dimy, numberMines)
    b = modMine(a)
    print(b)
    
main()
    
## (0~8 : number of mines nearby)
## (9 : mine)