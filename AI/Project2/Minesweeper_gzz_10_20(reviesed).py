#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:38:45 2017

@author: zhenzhenge
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


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
            

def exploreNeighbors(neighbors):
    stats = {i: [] for i in ["unknown","clue","mine","clear"]}
    for neighbor in neighbors:
        if neighbors[neighbor]==25:
            stats["clear"].append(neighbor)
        if neighbors[neighbor]==15:
            stats["mine"].append(neighbor)
        if int(neighbors[neighbor])<=8 and int(neighbors[neighbor])>=0:
            stats["clue"].append(neighbor)
        if neighbors[neighbor]==-10:
            stats["unknown"].append(neighbor)
    return stats

    
def getGrid(dim,mineNum):
    arr = np.append(np.repeat("M",mineNum),np.repeat(0,dim**2-mineNum))
    np.random.shuffle(arr)
    grid = arr.reshape(dim,dim)
    trackTable = np.repeat(-10,dim**2).reshape(dim,dim)
    
    for row in range(dim):
        for col in range(dim):
            if grid[row,col] == "M":
                continue
            else:
                grid[row,col]=list(getNeighbors(grid,row,col).values()).count("M")
    return grid, trackTable

def showGrid(trackTable):
    
    dim = len(trackTable)
    # white-known and not mine: 0~8; 
    # grey-unknown: -10;
    # red-marked as mine: 15;
    # green-recommended to query: 25
    cmap = colors.ListedColormap(['grey','white','red','green'])
    bounds = [-11,-1,9,19,29]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(figsize=(10, 10))
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
            elif trackTable[row,col] == 18:
                plt.text(col,row,"*")
            elif trackTable[row,col] == 25:
                plt.text(col,row,"C")
            else:
                plt.text(col,row,"?")
    for i in range(dim):
        plt.text(i,-1,i)
        plt.text(-1,i,i)
                
    
    plt.show()
    
    
def queryCell(grid,trackTable,row,col):

    if grid[row,col] == "M":
        trackTable[row,col] = 18
        return False,trackTable
    else:
        trackTable[row,col] = grid[row,col]
        return True,trackTable
    

 
def markCell(trackTable,row,col,markVal):

    if trackTable[row,col] == -10:
        trackTable[row,col] = markVal
    
    return trackTable


def simpleLogic(grid,trackTable):

    dim = len(trackTable)
    visited = set()
    for row in range(dim):
        for col in range(dim):

            if (trackTable[row,col] == -10 or trackTable[row,col] == 15 
                or (row,col) in visited):
                continue

            visited.add((row,col))
            neighbors = getNeighbors(trackTable,row,col)
            stats = exploreNeighbors(neighbors)
            mineNeighbor = stats["mine"]
            unknownNeighbor = stats["unknown"] 
            
            if trackTable[row,col] == 25:
                proceed, trackTable = queryCell(grid,trackTable,row,col)
                if not proceed:
                    return proceed, trackTable   
                
            if trackTable[row,col] >= 0 and trackTable[row,col] <= 8:
                if trackTable[row,col] == len(mineNeighbor):
                    for neighbor in unknownNeighbor:
                        trackTable = markCell(trackTable,neighbor[0],neighbor[1],25)
                        visited.add(neighbor)
                elif trackTable[row,col] - len(mineNeighbor) == len(unknownNeighbor):
                    for neighbor in unknownNeighbor:
                        trackTable = markCell(trackTable,neighbor[0],neighbor[1],15) 
                        visited.add(neighbor)
    
    return True, trackTable


def simpleLogicForTry(trackTable):

    dim = len(trackTable)
    visited = set()
    for row in range(dim):
        for col in range(dim):

            if (trackTable[row,col] in [-10,15,25] or (row,col) in visited):
                continue

            visited.add((row,col))
            neighbors = getNeighbors(trackTable,row,col)
            stats = exploreNeighbors(neighbors)
            mineNeighbor = stats["mine"]
            unknownNeighbor = stats["unknown"] 
            
              
            if trackTable[row,col] >= 0 and trackTable[row,col] <= 8:
                if trackTable[row,col] == len(mineNeighbor):
                    for neighbor in unknownNeighbor:
                        trackTable = markCell(trackTable,neighbor[0],neighbor[1],25)
                        visited.add(neighbor)
                elif trackTable[row,col] - len(mineNeighbor) == len(unknownNeighbor):
                    for neighbor in unknownNeighbor:
                        trackTable = markCell(trackTable,neighbor[0],neighbor[1],15) 
                        visited.add(neighbor)
    
    return trackTable
                
    
def tryAndError(trackTable):
    copyTable = np.array(trackTable)
    unknownList = np.argwhere(copyTable == -10)
    for cell in unknownList:
        copyTable[cell[0],cell[1]] = 15
        isMine = detectError(simpleLogicForTry(np.array(copyTable)))
        copyTable[cell[0],cell[1]] = 25
        if not isMine:
            trackTable[cell[0],cell[1]] = 25           
        else:       
            isNotMine = detectError(simpleLogicForTry(np.array(copyTable)))
            if not isNotMine:
                trackTable[cell[0],cell[1]] = 15
                copyTable[cell[0],cell[1]] = 15
            else:
                copyTable[cell[0],cell[1]] = -10
            
           
    return trackTable
    

   
 
def detectError(trackTable):
    dim = len(trackTable)
    for row in np.arange(dim):
        for col in np.arange(dim):
            if trackTable[row,col] in [-10,15,25]:
                continue
            neighbors = getNeighbors(trackTable,row,col)
            stats = exploreNeighbors(neighbors)
            mineNeighbor = stats["mine"]
            unknownNeighbor = stats["unknown"] 
            if ((trackTable[row,col]-len(mineNeighbor) > len(unknownNeighbor)) or 
                (trackTable[row,col]-len(mineNeighbor) < 0)): 
                return False
    return True

def getRandomUnknown(trackTable):
    toQuery = np.argwhere(trackTable == -10)
    cell = toQuery[np.random.randint(len(toQuery))]
    return cell[0], cell[1]


def mineSweeper(dim,mine):
    

    grid,trackTable = getGrid(dim,mine) 
    row,col = getRandomUnknown(trackTable)
    proceed, trackTable = queryCell(grid,trackTable,row,col) 
    #showGrid(trackTable)
    left = mine
    while proceed and len(np.argwhere(trackTable == 15)) < mine:
        unknown = len(np.argwhere(trackTable == -10))
        proceed, trackTable = simpleLogic(grid,trackTable)
        if not proceed:            
            print("Game Over!")
            break
        new_unknown = len(np.argwhere(trackTable == -10))
        if new_unknown == unknown:
            trackTable = tryAndError(trackTable)
            newer_unknown = len(np.argwhere(trackTable == -10))
            if newer_unknown == new_unknown:
                row,col = NextProb(trackTable,left)
                proceed, trackTable = queryCell(grid,trackTable,row,col)
                if not proceed:
                   print("Game Over!")
                   break
        showGrid(trackTable)
        left = mine-len(np.argwhere(trackTable == 15))
        print("There are",left,"mines left!")
    if left == 0 and -10 in trackTable:
        for cell in np.argwhere(trackTable == -10):
            trackTable[cell[0],cell[1]]=25
    showGrid(trackTable)

def NextProb(trackTable, remainMine):
    
    prob = dict()

    for i in range(len(trackTable)):
        for j in range(len(trackTable[0])):
            if trackTable[i,j] != -10:
                neighbors = getNeighbors(trackTable,i,j)
                mineNeighbor = list(neighbors.values()).count(15)
                unknownNeighbor = list(neighbors.values()).count(-10)
                for key in neighbors.keys():
                    if trackTable[key] == -10:
                        if key in prob:
                            prob[key] = max(prob[key], (int(trackTable[i,j]) - mineNeighbor)/unknownNeighbor)
                        else:
                            prob[key] = (int(trackTable[i,j]) - mineNeighbor)/unknownNeighbor
            else : 
                neighbors = getNeighbors(trackTable,i,j)
                if min(neighbors.values()) == max(neighbors.values()):
                    prob[key] = remainMine/((trackTable==-10).sum())
            
    min = 1
    minKey = 0
    for key in prob:
        if prob[key] < min:
            min = prob[key]
            minKey = key
            
    return int(minKey[0]), int(minKey[1])

mineSweeper(10,12)




            
