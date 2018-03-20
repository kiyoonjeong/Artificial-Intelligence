#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:38:45 2017

@author: zhenzhenge
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import random

def getNeighbors(trackTable,row,col):

    
    dim = len(trackTable)
    neighbors = {}
    for i in range(max(0,row-1),min(dim,row+2)):
        for j in range(max(0,col-1),min(dim,col+2)):
            if i == row and j == col:
                continue
            else:
                neighbors[(i,j)] = trackTable[i,j]
    return neighbors
            
    
def getGrid(dim,mineNum):
    arr = np.append(np.repeat("M",mineNum),np.repeat(0,dim**2-mineNum))
    np.random.shuffle(arr)
    grid = arr.reshape(dim,dim)
    trackTable = np.repeat(-10,dim**2).reshape(dim,dim)
    probTable = np.repeat(mineNum/dim**2, dim**2).reshape(dim,dim)
    
    for row in range(dim):
        for col in range(dim):
            if grid[row,col] == "M":
                continue
            else:
                grid[row,col]=list(getNeighbors(grid,row,col).values()).count("M")
    return grid, trackTable, probTable

def queryCell(grid,trackTable,row,col):

    if grid[row,col] == "M":
        return "Game Over!"
    else:
        trackTable[row,col] = grid[row,col]
        return trackTable
    


def exploreNeighbors(neighbors):
    stats = {i: [] for i in ["unknown","clue","mine","clear"]}
    for neighbor in neighbors:
        if neighbors[neighbor]==25:
            stats["clear"].append(neighbor)
        if neighbors[neighbor]==15:
            stats["mine"].append(neighbor)
        if neighbors[neighbor]<=8 and neighbors[neighbor]>=0:
            stats["clue"].append(neighbor)
        if neighbors[neighbor]==-10:
            stats["unknown"].append(neighbor)
    return stats

    

def mineSweeper(grid,trackTable, probTable):

    dim = len(trackTable)
    unknown = list(trackTable.flatten()).count(-10)
    remainMine = list(grid.flatten()).count("M") - list(trackTable.flatten()).count(15)
    visited = set()
    for row in range(dim):
        for col in range(dim):

            if trackTable[row,col] == -10 or trackTable[row,col] == 15:
                continue
            
            if (row,col) in visited:
                continue
            visited.add((row,col))
            neighbors = getNeighbors(trackTable,row,col)
            mineNeighbor = list(neighbors.values()).count(15)
            unknownNeighbor = list(neighbors.values()).count(-10)
            
            if trackTable[row,col] == 25:
                trackTable = queryCell(grid,trackTable,row,col)
    ####
                for key in neighbors.keys():
                    if grid[row,col] == mineNeighbor:
                        probTable[key] = 0
                    elif unknownNeighbor == 0 :
                        probTable[key] = 0
                    else : 
                        probTable[key] = max(probTable[key], (int(grid[row,col]) - mineNeighbor)/unknownNeighbor)
                
                if trackTable == "Game Over!":
                    return trackTable
                
            if trackTable[row,col] >= 0 and trackTable[row,col] <= 8:
                if trackTable[row,col] == len(neighbors):
                    for neighbor in neighbors:
                        trackTable[neighbor[0],neighbor[1]] = 15

                else:
                    if trackTable[row,col] == mineNeighbor:
                        for neighbor in neighbors:
                            if trackTable[neighbor[0],neighbor[1]] == -10 :
                                trackTable[neighbor[0],neighbor[1]] = 25
                                visited.add((neighbor[0],neighbor[1]))
                    elif trackTable[row,col] > mineNeighbor:
                        if trackTable[row,col]-mineNeighbor == unknownNeighbor:
                            for neighbor in neighbors:
                                if trackTable[neighbor[0],neighbor[1]] == -10 :
                                    trackTable[neighbor[0],neighbor[1]] = 15
                                    visited.add((neighbor[0],neighbor[1]))
    

    if list(trackTable.flatten()).count(-10) == unknown:
        minval = np.min(probTable[np.nonzero(probTable)])
        Test = np.argwhere(probTable == minval)
        Try = random.choice(Test)
        row = Try[0]
        col = Try[1]
        trackTable = queryCell(grid,trackTable,row,col)
        
        neighbors = getNeighbors(trackTable,row,col)
        mineNeighbor = list(neighbors.values()).count(15)
        unknownNeighbor = list(neighbors.values()).count(-10)
                
        for key in neighbors.keys():
            if grid[row,col] == mineNeighbor:
                probTable[key] = 0
            elif unknownNeighbor == 0 :
                probTable[key] = 0
            else : 
                probTable[key] = max(probTable[key], (int(grid[row,col]) - mineNeighbor)/unknownNeighbor, remainMine/unknown)
                
                                        
    return trackTable
                
    
                
    

def showGrid(trackTable):
    
    if trackTable == "Game Over!":
        return trackTable
    dim = len(trackTable)
    # white-known and not mine: 0~8; 
    # grey-unknown: -10;
    # red-marked as mine: 15;
    # green-recommended to query: 25
    cmap = colors.ListedColormap(['grey','white','red','green'])
    bounds = [-11,-1,9,19,29]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(trackTable, cmap=cmap, norm=norm)
    
    # draw gridlines
    ax.grid(which = 'major', axis = 'both', linestyle='-', color='k', linewidth=0.5)
    ax.set_xticks(np.arange(-0.5, dim, 1));
    ax.set_yticks(np.arange(-0.5, dim, 1));
    
    # Remove axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add texts
    for row in range(dim):
        for col in range(dim):
            if trackTable[row,col] >= 0 and trackTable[row,col] <= 8:
                plt.text(col,row,trackTable[row,col])
            elif trackTable[row,col] == 15:
                plt.text(col,row,"M")
            elif trackTable[row,col] == 25:
                plt.text(col,row,"C")
            else:
                plt.text(col,row,"?")
    for i in range(dim):
        plt.text(i,-1,i+1)
        plt.text(-1,i,i+1)
                
            

    
    plt.show()
                
    

     
grid,trackTable, probTable = getGrid(5,10) 
showGrid(trackTable)
while -10 in trackTable:    
    trackTable = mineSweeper(grid,trackTable, probTable)
    if trackTable == "Game Over!":
        print(trackTable)
        break
    showGrid(trackTable)







