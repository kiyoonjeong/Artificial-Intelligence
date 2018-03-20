#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 23:26:49 2017

@author: zhenzhenge
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import colors
from collections import deque
from heapq import heappop, heappush


def getMaze():   
    maze = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1],
            [1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1],
            [1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,1,1,0,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1],
            [1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,1],
            [1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,1,0,1,0,1],
            [1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,1,0,1],
            [1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1],
            [1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
            ]
    return maze
    

def getMaze1():
    
    dim = int(input("Enter maze dimension: "))
    ratio = float(input("Enter p: "))
    
    # Generate a binary value matrix with certain proportion of value 1
    maze = np.random.choice(2, size=(dim, dim),p=[1-ratio,ratio])
    maze[0,0] = 0
    maze[len(maze)-1,len(maze)-1] = 0
    
    return maze
    
def getMaze1(dim,ratio):
    
    dim = int(dim)
    ratio = float(ratio)
    
    # Generate a binary value matrix with certain proportion of value 1
    maze = np.random.choice(2, size=(dim, dim),p=[1-ratio,ratio])
    maze[0,0] = 0
    maze[len(maze)-1,len(maze)-1] = 0
    
    return maze

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
    return graph






def getBFS(maze):
    start, goal = (0,0), (len(maze) - 1, len(maze) - 1)
    queue = deque([("", start)])
    visited = set()
    graph = getGraph(maze)
    while queue:
        path, current = queue.popleft()
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            queue.append((path + direction, neighbour))
    return "Fail to find a path."
    
def getBFS1(maze):
    start, goal = (0,0), (len(maze) - 1, len(maze) - 1)
    queue = deque([("", start)])
    visited = set()
    graph = getGraph(maze)
    while queue:
        path, current = queue.popleft()
        if current == goal:
            return visited
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            queue.append((path + direction, neighbour))
    return "Fail to find a path."
    
def getBFS2(maze):
    start, goal = (0,0), (len(maze) - 1, len(maze) - 1)
    queue = deque([("", start)])
    visited = set()
    graph = getGraph(maze)
    while queue:
        path, current = queue.popleft()
        if current == goal:
            return visited
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            queue.append((path + direction, neighbour))
    return visited
    
def getBFS3(maze):
    start, goal = (0,0), (len(maze) - 1, len(maze) - 1)
    queue = deque([("", start)])
    visited = set()
    graph = getGraph(maze)
    maxq = len(queue)
    while queue:
        if maxq < len(queue):
            maxq = len(queue)
        path, current = queue.popleft()
        if current == goal:
            return maxq
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
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            stack.append((path + direction, neighbour))
    return "Fail to find a path."

def getDFS1(maze):
    
    start, goal = (0,0), (len(maze) - 1, len(maze) - 1)
    stack = deque([("", start)])
    visited = set()
    graph = getGraph(maze)
    while stack:
        path, current = stack.pop()
        if current == goal:
            return visited
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            stack.append((path + direction, neighbour))
    return "Fail to find a path."    

def getDFS3(maze):
    
    start, goal = (0,0), (len(maze) - 1, len(maze) - 1)
    stack = deque([("", start)])
    visited = set()
    graph = getGraph(maze)
    maxs = len(stack)
    while stack:
        if maxs < len(stack):
            maxs = len(stack)
        path, current = stack.pop()
        if current == goal:
            return maxs
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
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicEuclidean(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "Fail to find a path."

def getAStar_Euclidean1(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicEuclidean(start, goal), 0, "", start))
    visited = set()
    graph = getGraph(maze)
    while priority_queue:
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return visited
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicEuclidean(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "Fail to find a path."
    
def getAStar_Euclidean2(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicEuclidean(start, goal), 0, "", start))
    visited = set()
    graph = getGraph(maze)
    while priority_queue:
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return visited
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicEuclidean(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return visited

def getAStar_Euclidean3(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicEuclidean(start, goal), 0, "", start))
    visited = set()
    maxpq = len(priority_queue)
    graph = getGraph(maze)
    while priority_queue:
        if maxpq < len(priority_queue):
            maxpq = len(priority_queue)
            print(priority_queue)
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return maxpq
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicEuclidean(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "Fail to find a path."

def getAStar_Euclidean4(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicEuclidean(start, goal), 0, "", start))
    visited = set()
    maxpq = priority_queue
    graph = getGraph(maze)
    while priority_queue:
        if len(maxpq) < len(priority_queue):
            maxpq = priority_queue
            print(len(priority_queue))
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return maxpq
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
            return path
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicManhattan(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "Fail to find a path."

def getAStar_Manhattan1(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicManhattan(start, goal), 0, "", start))
    visited = set()
    graph = getGraph(maze)
    while priority_queue:
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return visited
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicManhattan(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "Fail to find a path."
    
def getAStar_Manhattan2(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicManhattan(start, goal), 0, "", start))
    visited = set()
    graph = getGraph(maze)
    while priority_queue:
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return visited
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicManhattan(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return visited

def getAStar_Manhattan3(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicManhattan(start, goal), 0, "", start))
    visited = set()
    maxpq = len(priority_queue)
    graph = getGraph(maze)
    while priority_queue:
        if maxpq < len(priority_queue):
            maxpq = len(priority_queue)
            print(priority_queue)
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return maxpq
        if current in visited:
            continue
        visited.add(current)
        for direction, neighbour in graph[current]:
            heappush(priority_queue, (cost + heuristicManhattan(neighbour, goal), cost + 1,
                                path + direction, neighbour))
    return "Fail to find a path."

    
def getAStar_Manhattan4(maze):

    start, goal = (0, 0), (len(maze) - 1, len(maze) - 1)
    priority_queue = []
    heappush(priority_queue, (0 + heuristicManhattan(start, goal), 0, "", start))
    visited = set()
    maxpq = priority_queue
    graph = getGraph(maze)
    while priority_queue:
        if len(maxpq) < len(priority_queue):
            maxpq = priority_queue
            print(len(maxpq))
        _, cost, path, current = heappop(priority_queue)
        if current == goal:
            return maxpq
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
        
                

def main():
    
    maze = getMaze()
    showMaze(maze)
    print("")
    
    bfsMaze = np.array(maze)
    start_time = time.time()
    bfs = getBFS(bfsMaze)
    print("--- %s seconds ---" % (time.time() - start_time))

    
    print("BFS Algorithm")
    showPath(bfsMaze, bfs)
    print("")
    
    dfsMaze = np.array(maze)
    start_time = time.time()
    dfs = getDFS(dfsMaze)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("DFS Algorithm")
    showPath(dfsMaze, dfs)
    print("")
    
    astarEuclideanMaze = np.array(maze)
    start_time = time.time()
    astarEuclidean = getAStar_Euclidean(astarEuclideanMaze)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Astar Algorithm with Euclidean Distance")
    showPath(astarEuclideanMaze, astarEuclidean)
    print("")
    
    astarManhattanMaze = np.array(maze)
    start_time = time.time()
    astarManhattan = getAStar_Manhattan(astarManhattanMaze)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Astar Algorithm with Manhattan Distance")
    showPath(astarManhattanMaze, astarManhattan)

def main1():
    i = 0.1
    while i < 0.95:
        count = 0
        b=0
        d=0
        m=0
        e=0
        for j in range(0,10):
            maze = getMaze1(300,i)        
            bfsMaze = np.array(maze)
            start_time = time.time()
            bfs = getBFS(bfsMaze)
            b += time.time() - start_time 
            if bfs == "Fail to find a path.":
                count += 1
            dfsMaze = np.array(maze)
            start_time = time.time()
            dfs = getDFS(dfsMaze)
            d += time.time() - start_time
            astarEuclideanMaze = np.array(maze)
            start_time = time.time()
            astarEuclidean = getAStar_Euclidean(astarEuclideanMaze)
            e += time.time() - start_time
            astarManhattanMaze = np.array(maze)
            start_time = time.time()
            astarManhattan = getAStar_Manhattan(astarManhattanMaze)
            m += time.time() - start_time
        print(count,b,d,m,e)
        count = 0
        b = 0
        d = 0
        m = 0
        e = 0
        i= i+0.1

def main2():
    i = 0.1
    while i < 0.35:
        n = 0
        b = 0
        d = 0
        m = 0
        e = 0
        while n < 100:
            maze = getMaze1(300,i)
            bfsMaze = np.array(maze)
            bfs = getBFS(bfsMaze)
            if bfs != "Fail to find a path.":
                n += 1
                b += len(bfs) 
    
            dfsMaze = np.array(maze)
            dfs = getDFS(dfsMaze)
            if dfs != "Fail to find a path.":
                d += len(dfs)

            astarEuclideanMaze = np.array(maze)
            astarEuclidean = getAStar_Euclidean(astarEuclideanMaze)
            if astarEuclidean != "Fail to find a path.":
                e += len(astarEuclidean)
    
            astarManhattanMaze = np.array(maze)
            astarManhattan = getAStar_Manhattan(astarManhattanMaze)
            if astarManhattan != "Fail to find a path.":
                m += len(astarManhattan)
        print(b/n,d/n,m/n,e/n)
        b = 0
        d = 0
        m = 0
        e = 0        
        i += 0.1


def main3():
    i = 0.1
    while i < 0.35:
        n = 0
        b = 0
        d = 0
        m = 0
        e = 0
        while n < 30:
            maze = getMaze1(300,i)
            bfsMaze = np.array(maze)
            bfs = getBFS1(bfsMaze)
            if bfs != "Fail to find a path.":
                n += 1
                b += len(bfs) 
    
            dfsMaze = np.array(maze)
            dfs = getDFS1(dfsMaze)
            if dfs != "Fail to find a path.":
                d += len(dfs)

            astarEuclideanMaze = np.array(maze)
            astarEuclidean = getAStar_Euclidean1(astarEuclideanMaze)
            if astarEuclidean != "Fail to find a path.":
                e += len(astarEuclidean)
    
            astarManhattanMaze = np.array(maze)
            astarManhattan = getAStar_Manhattan1(astarManhattanMaze)
            if astarManhattan != "Fail to find a path.":
                m += len(astarManhattan)
        print(b/n,d/n,m/n,e/n)
        b = 0
        d = 0
        m = 0
        e = 0        
        i += 0.1

# main1()  *dim = 100, *p = 0.1 ~ 0.9, *print -> number of failure, time consume for bfs, dfs, euclidean, and manhattan

# main2() *dim = 30, *p = 0.1 ~ 0.3, *print -> average of path (iterate n(1000) times)
# -> use len(path) instead of print(path)

# main3() *dim = 100, *p = 0.1 ~ 0.3 *print -> average of expanded nodes (iterate n(100) times)
# -> use visited instead of path

def main4():
    i = 0.4
    while i < 1.0:
        n = 0
        m = 0
        e = 0
        while n < 50:
            maze = getMaze1(300,i)
            astarEuclideanMaze = np.array(maze)
            astarEuclidean = getAStar_Euclidean2(astarEuclideanMaze)
            e += len(astarEuclidean)
    
            astarManhattanMaze = np.array(maze)
            astarManhattan = getAStar_Manhattan2(astarManhattanMaze)
            m += len(astarManhattan)
            n += 1
        print(m/n,e/n)
        m = 0
        e = 0        
        i += 0.1


def hardMaze1(dim, ratio):
    maze = getMaze1(dim, ratio)
    while getBFS(maze) == "Fail to find a path." :
        maze = getMaze1(dim, ratio)
    path = getBFS(maze)
    j=1
    x=0
    y=0

    while j < len(path):        
        for i in range(j):
            if path[i] == "S":
                x += 1
            if path[i] == "E":
                y += 1
            if path[i] == "N":
                x -= 1
            if path[i] == "W" :
                y -= 1
        maze[x][y] = 1
        if getBFS(maze) == "Fail to find a path." :
            maze[x][y] = 0
            j += 1
            x=0
            y=0
        else:
            path = getBFS(maze)
            j = 1
            x = 0
            y = 0
            
    return maze
    
def hardMaze2(dim, ratio):
    maze = getMaze1(dim, ratio)
    while getBFS(maze) == "Fail to find a path." :
        maze = getMaze1(dim, ratio)
        
    visited = getBFS1(maze)

#1. remove block 
#2. add block
    block = [] 
    for i in range(len(maze)):
        for j in range(len(maze[i])):
            if maze[i][j] == 1:
                block.append([i,j])
    shortest = [dim, dim]
    for dot in block:
        if sum(dot) < sum(shortest):
            shortest = dot
    maze[shortest[0]][shortest[1]] = 0
    if len(getBFS1(maze)) > len(visited):
        visited = getBFS1(maze)
    else:
        maze[shortest[0]][shortest[1]] = 1
        block
                
            
        

            
    return maze    
    
    

            




