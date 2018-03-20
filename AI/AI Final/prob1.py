#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:26:49 2017

@author: Kiyoon Jeong
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

#Maze
def getMaze():   
    maze = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
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
            ])
    return maze   

#Prob1    
def points(maze):
    zero = []
    block = []
    for i in range(37):
        for j in range(37):
            if maze[i,j] == 0:
                zero.append([i,j])
            else:
                block.append([i,j])
    return zero, block

#Move
       
def left(point):
    if maze[point[0],point[1]-1] == 0:
        return [point[0], point[1]-1]
    return [point[0], point[1]]

def right(point):
    if maze[point[0],point[1]+1] == 0:
        return [point[0], point[1]+1]
    return [point[0], point[1]]

def up(point):
    if maze[point[0]-1,point[1]] == 0:
        return [point[0]-1, point[1]]
    return [point[0], point[1]]

def down(point):
    if maze[point[0]+1,point[1]] == 0:
        return [point[0]+1, point[1]]
    return [point[0], point[1]]

def nleft(point, n):
    while n != 0:
        point = left(point)
        n -= 1
    return point

def nright(point, n):
    while n != 0:
        point = right(point)
        n -= 1
    return point
def ndown(point, n):
    while n != 0:
        point = down(point)
        n -= 1
    return point
def nup(point, n):
    while n != 0:
        point = up(point)
        n -= 1        
    return point
    
#Prob2
def main1():
    status = []
    success = 0
    for i in range(len(zero)):
        point = zero[i]  
        
        # phase 1 : get_out from the 1st square
        point = nright(point, 2)
        point = nleft(point,1)
        point = ndown(point,4)  
        
        # phase 2 : get_out from the 2nd square
        point = nright(point,6)    
        point = nup(point,6) 
        point = nright(point,6)
        point = nleft(point,3)
        point = nup(point,2)
        
        # phase 3 : arrived at the district9 & aggregate in one point
        point = nright(point,10)
        point = ndown(point,10)
        point = nleft(point,10)
        point = ndown(point, 24)
        point = nup(point, 10)
        point = nright(point,34)
        
        # phase 4 : into the goal
        point = nleft(point,5)
        point = ndown(point,2)
        point = nright(point,3)
        point = ndown(point,6)  
        point = nleft(point,3) 
        point = nup(point,3)

        status.append(point)
        if point == goal:
            success += 1
                    
    return status, success/len(zero)
#Total move : 150

#Prob3
def main2():
    status = []
    success = 0
    
    for i in range(len(zero)):
        point = zero[i]
        
        # phase 1 : get_out from the 1st square
        point = nright(point,2)
        point = nleft(point,1)
        point = ndown(point,4)
        
        # phase 2 : get_out from the 2nd square
        point = ndown(point,1)
        point = nright(point,6)
        point = nup(point,6)
        point = nleft(point,3)
        point = nup(point,2)
        
        #extra : if we up(point), then we can aggregate all of the left-most side
        point = up(point)
        
        # phase 3 :  aggregate in one point & arrive at district9
        point = nright(point,34)
        point = ndown(point, 34)
        point = nleft(point,10)
        point = ndown(point, 24)
        point = nup(point, 10)
        point = nright(point,34)
        
        # phase 4 : into the goal
        point = nleft(point,5)
        point = ndown(point,2)
        point = nright(point,3)
        point = ndown(point,6)  
        point = nleft(point,3) 
        point = nup(point,3)

        status.append(point)
        if point == goal:
            success += 1
        
                    
    return status, success/len(zero)
# Total move : 194



#Display    
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



# Points Location   
def showStatus(maze, status):        
    for i in range(len(status)):
        maze[status[i][0],status[i][1]] = 2
    showMaze(maze)    


#prob4
    
#number of surrounded blocks     
def nmap(maze):
    map = np.array([[0]*37]*37)
    for i in range(37):
        for j in range(37):
            if maze[i,j] == 0:
                map[i,j] = maze[i-1,j-1] +maze[i-1,j]+maze[i-1,j+1] + maze[i,j-1] + maze[i,j+1] + maze[i+1,j-1] +maze[i+1,j]+maze[i+1,j+1]
    
            else:
                map[i,j] = -1
    return map

#First Observation
def initcell(obs):
    possible_cell = []
    for i in range(37):
        for j in range(37):
            if map[i,j] == obs:
                possible_cell.append([i,j])
    return possible_cell
    
#possible points by using observations
def guesscell(obs, dir, list):
    filter = []
    possible_cell = initcell(obs)
    for i in range(len(list)):
        if dir == "l":
            if maze[list[i][0], list[i][1]-1] == 0:
                if [list[i][0], list[i][1]-1] in possible_cell:
                    filter.append([list[i][0],list[i][1]-1])
            else:
                if [list[i][0], list[i][1]] in possible_cell:
                    filter.append([list[i][0],list[i][1]])
        if dir == "r":
            if maze[list[i][0], list[i][1]+1] == 0:
                if [list[i][0], list[i][1]+1] in possible_cell:
                    filter.append([list[i][0],list[i][1]+1])
            else:
                if [list[i][0], list[i][1]] in possible_cell:
                    filter.append([list[i][0],list[i][1]])
        if dir == "u":
            if maze[list[i][0]-1, list[i][1]] == 0:
                if [list[i][0]-1, list[i][1]] in possible_cell:
                    filter.append([list[i][0]-1,list[i][1]])
            else:
                if [list[i][0], list[i][1]] in possible_cell:
                    filter.append([list[i][0],list[i][1]])
        if dir == "d":
            if maze[list[i][0]+1, list[i][1]] == 0:
                if [list[i][0]+1, list[i][1]] in possible_cell:
                    filter.append([list[i][0]+1,list[i][1]])
            else:
                if [list[i][0], list[i][1]] in possible_cell:
                    filter.append([list[i][0],list[i][1]])    
    return filter     
    

#print the most possible cell location
def maxprob(lists):
    prob = np.array([[0]*37]*37)
    for i in range(len(lists)):
        prob[lists[i][0]][lists[i][1]] += 1
    i,j = np.where(prob == np.max(prob))
    maxlist = []
    for k in range(len(i)):
        maxlist.append([i[k], j[k]])
               
    return maxlist



maze = getMaze()
goal = [30,30]
zero, block = points(maze)
print(len(zero))

status, rate = main1()
showStatus(maze, status)
print(rate)

maze = getMaze()
goal = [30,30]
zero, block = points(maze)

status2, rate2 = main2()
showStatus(maze, status2)
print(rate2)

maze = getMaze()
goal = [30,30]
zero, block = points(maze)
status3, rate3 = main2()
map = nmap(maze)

start_point = initcell(5)
nextpoint = guesscell(5,"l",start_point)
nextpoint2 = guesscell(5,"l",nextpoint)
print(maxprob(nextpoint2))

start_point2 = initcell(5)
testpoint = guesscell(6, "r", start_point2)
testpoint2 = guesscell(6, "u", testpoint)
print(maxprob(testpoint2))

