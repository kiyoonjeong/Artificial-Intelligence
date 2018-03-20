#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 18:48:49 2017

@author: xingxiong
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
from heapq import heappop, heappush
import time
from timeit import timeit as tt
import re
import random
import copy
import sys   

sys.setrecursionlimit(1000000)

def getMaze():
    
    dim = int(input("Enter maze dimension: "))
    ratio = float(input("Enter p: "))
    
    # Generate a binary value matrix with certain proportion of value 1
    maze = np.random.choice(2, size=(dim, dim),p=[1-ratio,ratio])
    maze[0,0] = 0
    maze[len(maze)-1,len(maze)-1] = 0
    
    return(maze)

def getMaze1(dim,ratio):
    
    dim = int(dim)
    ratio = float(ratio)
    
    # Generate a binary value matrix with certain proportion of value 1
    maze = np.random.choice(2, size=(dim, dim),p=[1-ratio,ratio])
    maze[0,0] = 0
    maze[len(maze)-1,len(maze)-1] = 0
    
    return(maze)

def showMaze(maze):
       
    # create discrete colormap
    cmap = colors.ListedColormap(['white', 'black', 'green'])
    bounds=[-0.5,0.5,1.5,2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots()
    ax.imshow(maze, cmap=cmap, norm=norm)

    
    # draw gridlines
    ax.grid(which='major', axis = 'both', linestyle='-', color='k', linewidth=1.5)
    ax.set_xticks(np.arange(-0.5, len(maze), 1));
    ax.set_yticks(np.arange(-0.5, len(maze), 1));
    
    # Remove axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add start and goal texts
    plt.text(-1/6, 1/6, 'S')
    plt.text(len(maze)-7/6, len(maze)-5/6, 'G')
    
    plt.show()


def getGraph(maze):
    
    height = len(maze)
    width = len(maze[0])
    graph = {(i, j): [] for j in range(width) for i in range(height) if not maze[i][j]}
    for row, col in graph.keys():
        if row < height - 1 and not maze[row + 1][col]:
            graph[(row, col)].append(("S", (row + 1, col)))
            graph[(row + 1, col)].append(("N", (row, col)))
        if col < width - 1 and not maze[row][col + 1]:
            graph[(row, col)].append(("E", (row, col + 1)))
            graph[(row, col + 1)].append(("W", (row, col)))
    return(graph)






def getBFS(maze):
    start, goal = (0,0), (len(maze) - 1, len(maze) - 1)
    queue = deque([("", start)])
    visited = set()
    graph = getGraph(maze)
    while queue:
        path, current = queue.popleft()
        if current == goal:
            return(path,visited)
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            queue.append((path + direction, neighbour))
    return "Fail to find a path."


def getDFS(maze):
    
    start, goal = (0,0), (len(maze) - 1, len(maze) - 1)
    stack = deque([("", start)])
    visited = set()
    graph = getGraph(maze)
    while stack:
        path, current = stack.pop()
        if current == goal:
            return(path,visited)
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            stack.append((path + direction, neighbour))
    return "Fail to find a path."


def heuristicEuclidean(position, goal):
    return ((position[0] - goal[0])**2 + (position[1] - goal[1])**2)**(1/2.0)
    
def getAStar_Euclidean(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicEuclidean(start, goal), 0, "", start))
    visited = set()
    graph = getGraph(maze)
    while priority_queue:
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return(path,visited)
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicEuclidean(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "Fail to find a path."
    
def heuristicManhattan(position, goal):
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def getAStar_Manhattan(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicManhattan(start, goal), 0, "", start))
    visited = set()
    graph = getGraph(maze)
    while priority_queue:
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return(path,visited)
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicManhattan(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "Fail to find a path."




def showPath(maze, path):        
    
    print("Path: " + path)
    if path != "Fail to find a path.":
        print("Path Length: " + str(len(path)))       
        currentX = 0
        currentY = 0
        for i in range(len(path)-1):
            if path[i] == "S":
                currentX = currentX + 1
            if path[i] == "E":
                currentY = currentY + 1
            if path[i] == "W":
                currentY = currentY - 1
            if path[i] == "N":
                currentX = currentX - 1
            maze[currentX,currentY] = 2
        showMaze(maze)
        

def harderMaze(Maze,count,fail_time = 0,length = 0,recur_time = 0,del_op = 0,add_op = 0):
    print(sum(sum(Maze)))
    recur_time += 1
    print('now is',fail_time)
    print(del_op + add_op)
    
    if fail_time > 10 or del_op + add_op > 3000:
        return(Maze)
    else:
        bfsMaze = np.array(Maze)
        #dfsMaze = np.array(Maze)
        
        '''
        I only use bfs to judge the complexity
        '''        
        start = time.time()
        bfs = getBFS(bfsMaze)
        #dfs = getDFS(dfsMaze)
        time_cost = time.time() - start
        #length = len(bfs[0]) + len(dfs[0])  
        length = len(bfs[0])          
        Maze_origi = copy.deepcopy(Maze)
        #Maze_origi_0 = copy.deepcopy(Maze)
        if sum(sum(Maze)) <= 450:
            '''
            for those points which equal to 0 and have lots of 0 nearby,
            increase the possibility by increasing their appearance in the sample list
            '''
            sample_list1 = [(i,j) for i in range(len(Maze)) for j in range(len(Maze[0])) if Maze[i,j] == 0]
            sample_list2 = [(i,j) for i in range(1,len(Maze)-1) for j in range(1,len(Maze[0])-1) if Maze[i,j] == 0 and sum(sum(Maze[i-1:i+2,j-1:j+2])) < 4] * 3
            for node in random.sample(sample_list1 + sample_list2,max(5-fail_time,2)):    
                if node != (0,0):
                    Maze[node] = 1
            
            bfsMaze = np.array(Maze)
            del_op += 1
            #dfsMaze = np.array(Maze)
            start = time.time()
            bfs = getBFS(bfsMaze)
            #dfs = getDFS(dfsMaze)
            time_cost1 = time.time() - start  
            
            '''
            notice that the return value of getBFS function has been modified to a tuple
            '''
            
            if bfs[0] == "F":
                del_op -= 1
                print('Fail')
                time.sleep(0.1)
                return(harderMaze(Maze_origi,count,fail_time,length,recur_time,del_op,add_op))
            elif time_cost1 > time_cost or len(bfs[0])  > length:
                time.sleep(0.1)
                return(harderMaze(Maze,count + 1,0,len(bfs[0]),recur_time,del_op,add_op))
            else:
                
                '''
                this situation means changing 0 to 1 can't make the maze harder so we change 1 to 0 instead.
                '''
                time.sleep(0.1)
                sample_list1 = [(i,j) for i in range(len(Maze)) for j in range(len(Maze[0])) if Maze[i,j] == 1]
                sample_list2 = [(i,j) for i in range(1,len(Maze)-1) for j in range(1,len(Maze[0])-1) if Maze[i,j] == 1 and sum(sum(Maze[i-1:i+2,j-1:j+2])) >= 5] * 3
                for node in random.sample(sample_list1 + sample_list2,max(5-fail_time,2)):    
                    Maze[node] = 0
                bfsMaze = np.array(Maze)
                #dfsMaze = np.array(Maze_origi)
                start = time.time()
                bfs = getBFS(bfsMaze)
                #dfs = getDFS(dfsMaze)
                time_cost2 = time.time() - start  
                if time_cost2 > time_cost or len(bfs[0]) > length:
                    add_op += 1
                    time.sleep(0.1)
                    return(harderMaze(Maze,count + 1,0,len(bfs[0]),recur_time,del_op,add_op))
                del_op -= 1
                return(harderMaze(Maze_origi,count,fail_time + 1,len(bfs[0]),recur_time,del_op,add_op))
        else:
            sample_list1 = [(i,j) for i in range(len(Maze)) for j in range(len(Maze[0])) if Maze[i,j] == 1]
            sample_list2 = [(i,j) for i in range(1,len(Maze)-1) for j in range(1,len(Maze[0])-1) if Maze[i,j] == 1 and sum(sum(Maze_origi[i-1:i+2,j-1:j+2])) >= 5] * 3
            for node in random.sample(sample_list1 + sample_list2,max(5-fail_time,2)):    
                Maze[node] = 0
            bfsMaze = np.array(Maze)
            #dfsMaze = np.array(Maze_origi)
            start = time.time()
            bfs = getBFS(bfsMaze)
            #dfs = getDFS(dfsMaze)
            time_cost2 = time.time() - start  
            if time_cost2 > time_cost or len(bfs[0]) > length:
                add_op += 1
                time.sleep(0.1)
                return(harderMaze(Maze,count + 1,0,len(bfs[0]),recur_time,del_op,add_op))
            return(harderMaze(Maze_origi,count,fail_time + 1,len(bfs[0]),recur_time,del_op,add_op))
                
            
    


maze = getMaze1(30,0)
Maze = np.array(maze)
bfs = getBFS(Maze)
while bfs[0] == "F":
    maze = getMaze1(30,0.1)
    Maze = np.array(maze)
    bfs = getBFS(Maze)
new_maze = harderMaze(Maze,0)

print(222222)
bfsMaze = np.array(new_maze)
bfs = getBFS(bfsMaze)
print("BFS Algorithm")
showPath(bfsMaze, bfs[0])
print("")

dfsMaze = np.array(new_maze)
dfs = getDFS(dfsMaze)
print("DFS Algorithm")
showPath(dfsMaze, dfs[0])
print("")

astarEuclideanMaze = np.array(new_maze)
astarEuclidean = getAStar_Euclidean(astarEuclideanMaze)
print("Astar Algorithm with Euclidean Distance")
print(astarEuclidean[1:10])
showPath(astarEuclideanMaze, astarEuclidean[0])
print("")  




