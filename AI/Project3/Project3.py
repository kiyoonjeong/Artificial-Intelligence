#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:52:13 2017

@author: zhenzhenge
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# Construct a class to implement the project.
class Map:
    
    # We set up a few variables for the map: dim, map with different terrains, 
    # belief/found states etc.
    #
    def __init__(self, dim = 50):
        self._dim = dim
        self._map = self.getMap()
        
        # Keep the observation records
        self._obs = []
        # The prior belief for all cells is 1/2500
        self._belief = np.repeat(1/self._dim**2,
                                       self._dim**2).reshape(self._dim,self._dim)
        # This array will store the prob of finding target in each cell
        self._found = np.repeat(0.0,self._dim**2).reshape(self._dim,self._dim)
        self._uniform = np.repeat(1,self._dim**2).reshape(self._dim,self._dim)
        # The target is fixed at initial
        self._target = (np.random.randint(self._dim),
                        np.random.randint(self._dim))
        # This set is used to store the target history
        self._targetSet = set()
        
    # Generate the map according the setup requirements.
    #
    def getMap(self):
        
        # Here we use number to represent different terrains:
        # terrains = ['Flat','Hilly','Forest','Maze']
        # 1 represent "Flat"
        # 2 represent "Hilly"
        # 3 represent "Forest"
        # 4 represent "Maze of caves"           
        prob = [0.2,0.3,0.3,0.2]        
        arr = []
        for i in range(len(prob)):
            arr=np.append(arr,np.repeat(i+1,self._dim**2*prob[i]))
        np.random.shuffle(arr)              
        return arr.reshape((self._dim,self._dim))

    # Show the map if we want to see it directly from time to time
    #
    def showMap(self,showTarget=False):
        
        # 1 represent "Flat", and shown in white
        # 2 represent "Hilly", and shown in grey
        # 3 represent "Forest", and shown in green
        # 4 represent "Maze of caves", and shown in black
        cmap = colors.ListedColormap(['white','grey','green','black'])
        bounds = [0.5,1.5,2.5,3.5,4.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self._map, cmap=cmap, norm=norm)
        
        # draw gridlines
        ax.grid(which = 'major', axis = 'both', linestyle='-', color='k', 
                linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, self._dim, 1));
        ax.set_yticks(np.arange(-0.5, self._dim, 1));
        
        # Remove axis labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Mark the target
        if showTarget:
            plt.text(self._target[1],self._target[0],"X",color='red',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=10,fontweight="bold")
                    
        plt.show()

   
    # Search the chosen cell and try to find the target
    # The search result depends on whether the target is in this cell and 
    # also on certain probability given 
    # Return True if found, False if failed
    #
    def search(self, row, col):
        
        self._obs.append((row,col))
        
        notFoundProb = {1:0.1,2:0.3,3:0.7,4:0.9}
        prob = notFoundProb[self._map[row,col]]
                
        if ((row,col) == self._target and 
            np.random.choice((0,1),p=[prob,1-prob])):
            return True
        return False
    
    
    # With the search result, update our belief and found probability states
    # Use prior information to update posterior of our beliefs
    # Use belief information to update found probability
    #
    def update(self, row, col):
      
        notFoundProb = {1:0.1,2:0.3,3:0.7,4:0.9}
        
        for i in range(self._dim):
            for j in range(self._dim):
                cellProb = notFoundProb[self._map[i,j]]
                if (i,j) == (row,col):
                    self._belief[i,j] = cellProb*self._belief[i,j]
                self._found[i,j] = (1-cellProb)*self._belief[i,j]
        self._uniform[row,col] = 0
        norm = self._belief.sum()
        self._belief /= norm
        self._found /= norm
    

    # Choose the next cell to search by different rules according to the 
    # arguments passed by "by"
    #
    def getNext(self,by):
        
        if by == "belief":
            return np.argwhere(self._belief == np.max(self._belief))[0]
        if by == "found":
            return np.argwhere(self._found == np.max(self._found))[0]
        if by == "uniform":
            n = np.random.choice(len(np.argwhere(self._uniform == 1)),1)[0]
            #print(n,len(np.argwhere(self._uniform == 1)))
            return np.argwhere(self._uniform == 1)[n]
        
    
    # Get current target's terrain type
    #
    def getTargetType(self):
        terrains = {1:'Flat',2:'Hilly',3:'Forest',4:'Maze'}
        return terrains[self._map[self._target[0],self._target[1]]]
        
    
    # Use above functions to find target by designated rules passing by "by"
    # Report the steps taken before found
    #
    def findTarget(self,by):
        
        found = False
        while not found:
            goTo = self.getNext(by)
            found = self.search(goTo[0],goTo[1])
            self.update(goTo[0],goTo[1])
            if np.sum(self._uniform) == 0:
                self.clearRcords_for_uniform()
                return(self.findTarget(by))
        #print(self.getTargetType())
        return len(self._obs)
    
    
    # Reset current target cell without changing map
    #
    def resetTarget(self):
         
        chosen = False
        self._targetSet.add(self._target)
        while not chosen:
            newTarget = (np.random.randint(self._dim),np.random.randint(self._dim))          
            if newTarget not in self._targetSet:
                self._target = newTarget
                self._targetSet.add(self._target)
                chosen = True
    
    # Get neighbors' coordinates of this cell in up, down, left, right directions 
    #       
    def getNeighbors(self,row,col):
        neighbors = [(max(0,row-1),col),(min(self._dim-1,row+1),col),
                     (row,max(0,col-1)),(row,min(self._dim-1,col+1))]
        while (row,col) in neighbors:
            neighbors.remove((row,col))
        return neighbors
    

    # Move current target to one of the neighboring cell with uniform probability
    #         
    def moveTargetToNeighbor(self):
        neighbors = self.getNeighbors(self._target[0],self._target[1])
        self._target = neighbors[np.random.randint(len(neighbors))]

    # Clear previous records of belief/found/observations
    #
    def clearRecords(self):
        # Re-initialize the records
        self._obs = []
        self._belief = np.repeat(1/self._dim**2,
                                       self._dim**2).reshape(self._dim,self._dim)
        self._found = np.repeat(0.0,self._dim**2).reshape(self._dim,self._dim)
        self._uniform = np.repeat(1,self._dim**2).reshape(self._dim,self._dim)
    
    # A Clear function especially for uniform search.
    def clearRcords_for_uniform(self):
        self._uniform = np.repeat(1,self._dim**2).reshape(self._dim,self._dim)
        
    # Compare rule1 and rule2 on same map
    # Run several times and record the result sequence
    #
    def testMethodByRule(self,loop):
        # Calculate the total steps
        total1,total2 = [],[]
        
        for i in range(loop):
            # See target type
            print(self.getTargetType())
            # Use decision rule 1
            self.clearRecords()
            total1.append(self.findTarget(by="belief"))
            # Use decision rule 2
            self.clearRecords()
            total2.append(self.findTarget(by="found"))
            # Reset the target and test again
            self.resetTarget()
            
        return total1,total2
    
    # Compare rule1 and rule2 on multiple maps
    # Run several times and record the result sequence
    # 
    def testMethodWithMultiMap(self,loop):
        
        total1,total2 = [],[]
        
        for i in range(loop):
            # See target type
            print(self.getTargetType())
            # Use decision rule 1
            self.clearRecords()
            total1.append(self.findTarget(by="belief"))
            # Use decision rule 2
            self.clearRecords()
            total2.append(self.findTarget(by="found"))
            # Reset the target and test again
            self.resetTarget()
            self._map = self.getMap()
            
        return total1,total2
    
    
    # Choose next cell to search among all neighboring cells
    # Choose the one with highest belief
    #
    def getNextNeighbor(self,row,col):
        maxBelief = 0
        neighbors = self.getNeighbors(row,col)
        for (i,j) in neighbors:
            if self._belief[i,j] > maxBelief:
                maxBelief = self._belief[i,j]
                goTo = (i,j)
        return goTo
        
    # Always move to the "best" neighboring cell and try to find the target
    # Report the steps taken before found
    #
    def findTargetInNeighbors(self):
        
        goTo = (np.random.randint(self._dim),np.random.randint(self._dim))
        found = False
        while not found:
            goTo = self.getNextNeighbor(goTo[0],goTo[1])
            found = self.search(goTo[0],goTo[1])
            self.update(goTo[0],goTo[1])
        #print(self.getTargetType())
        return len(self._obs)
    
    # Always move to the "best" global cell and try to find the target
    # Report the steps taken before found plus moving distance in each step
    # The moving distance is calculated by Manhattan distance
    # 
    def findTargetByMove(self,by):
        
        found = False
        steps = 0
        current = (0,0)       
        while not found:           
            goTo = self.getNext(by)
            steps += abs(goTo[0]-current[0])+abs(goTo[1]-current[1])-1
            current = goTo
            found = self.search(goTo[0],goTo[1]) 
            self.update(goTo[0],goTo[1])
    
        #print(self.getTargetType())
        return len(self._obs)+steps    
    

    # Compare "always move to neighbor" and "always move by belief/found" 
    # Run several times and record the result sequence
    #    
    def testMethodWithRestrict(self,loop,by):
        
        total1,total2 = [],[]
        
        for i in range(loop):
            # See target type
            print(self.getTargetType())
            # Moving to neighbor
            self.clearRecords()
            total1.append(self.findTargetInNeighbors())
            # Use decision rule 1
            self.clearRecords()
            total2.append(self.findTargetByMove(by))
            # Reset the target and test again
            self.resetTarget()
            #self._map = self.getMap()
            
        return total1,total2
    
    # Compare "always move to neighbor","always move by belief",
    # "always move by found" on same map
    # Run several times and record the result sequence
    # 
    def testMethodWithRestrictAmongThree(self,loop):
        
        total1,total2,total3 = [],[],[]
        
        for i in range(loop):
            # See target type
            print(self.getTargetType())
            # Moving to neighbor
            self.clearRecords()
            total1.append(self.findTargetInNeighbors())
            # Use decision rule 1
            self.clearRecords()
            total2.append(self.findTargetByMove(by='belief'))
            # Use decision rule 1
            self.clearRecords()
            total3.append(self.findTargetByMove(by='found'))
            # Reset the target and test again
            self.resetTarget()
            #self._map = self.getMap()
            
        return total1,total2,total3
    
    # Address the questions in project Part Two
    # Update belief/found states when we have additional type1*type2 report
    # We'll find potential pool of the target cell and flip the probability 
    # between pairs to pass the information
    #     
    def updateWithMovingTarget(self,type1,type2):
               
        notFoundProb = {1:0.1,2:0.3,3:0.7,4:0.9}
        terrains = {1:'Flat',2:'Hilly',3:'Forest',4:'Maze'}

        pool = {}
        total = set()
        
        for row in range(self._dim):
            for col in range(self._dim):
                types = [type1,type2]
                currentType = terrains[self._map[row,col]]
                if currentType in types and self._belief[row,col]:
                    pool[(row,col)] = []
                    neighbors = self.getNeighbors(row,col)
                    types.remove(currentType)
                    for (i,j) in neighbors:
                        if terrains[self._map[i,j]] in types and (row,col) != (i,j):
                            pool[(row,col)].append((i,j))
                            total.add((row,col))
                            total.add((i,j))
               

        flipped = set()
        for key in pool:
            if len(pool[key]) > 0 and key not in flipped:
                temp = self._belief[key[0],key[1]]
                for keyitem in pool[key]:
                    if self._belief[keyitem[0],keyitem[1]] != temp and keyitem not in flipped:
                        self._belief[key[0],key[1]],self._belief[keyitem[0],keyitem[1]] = self._belief[keyitem[0],keyitem[1]],temp
                        flipped.add(key)
                        flipped.add(keyitem)
                    if self._belief[keyitem[0],keyitem[1]] == temp:
                        flip = True
                if flip:
                    self._belief[key[0],key[1]] = temp
                
                
        for m in range(self._dim):
            for n in range(self._dim):   
                if (m,n) not in total:
                    self._belief[m,n] = 0.0
                self._found[m,n] = (1-notFoundProb[self._map[m,n]])*self._belief[m,n]
        #print(self._belief)#debug
        norm = self._belief.sum()
#        if norm == 0.0:
#            return False
        self._belief /= norm
        self._found /= norm
#        return True

 
    # Use above functions to find target with additional type1*type2 report
    # by designated rules passing by "by"
    # Report the steps taken before found
    #     
    def findMovingTarget(self,by):
        
        type1 = self.getTargetType()
        goTo = self.getNext(by)
        found = self.search(goTo[0],goTo[1]) 
        
        while not found: 
            
            self.moveTargetToNeighbor()
            type2 = self.getTargetType()
            #print("The target was seen at a",oldType,"*",newType,"border.")
            self.updateWithMovingTarget(type1,type2)#debug
                #return len(self._obs) #debug
            type1 = type2
            goTo = self.getNext(by)
            found = self.search(goTo[0],goTo[1])
                        
        return len(self._obs)   
    
    
    # Under new setting, compare "always move by belief","always move by 
    # found probability" on the same map
    # Run several times and record the result sequence
    #    
    def testMethodWithMovingTarget(self,loop):
        
        total1,total2 = [],[]
        
        for i in range(loop):
            # See target type
            print(self.getTargetType())
            # Use decision rule 1
            self.clearRecords()
            total1.append(self.findMovingTarget(by="belief"))
            # Use decision rule 2
            self.clearRecords()
            total2.append(self.findMovingTarget(by="found"))
            # Reset the target and test again
            self.resetTarget()
            
        return total1,total2
    
    
    # Test average performance among Rule1, Rule2 and Uniform.
    # Test every type of target five times, get the average number of searches and total
    # weighted performance.
    def test_rule1_rule2_uniform(self,iter_time):
        flat_r1,hill_r1,forest_r1,cave_r1 = [],[],[],[]
        flat_r2,hill_r2,forest_r2,cave_r2 = [],[],[],[]
        flat_u,hill_u,forest_u,cave_u = [],[],[],[]
        dic_r1 = {"Flat":0,"Hilly":0,"Forest":0,"Maze":0}
        dic_r2 = {"Flat":0,"Hilly":0,"Forest":0,"Maze":0}
        dic_u = {"Flat":0,"Hilly":0,"Forest":0,"Maze":0}
        
        while sum(dic_r1.values()) < 4 * int(iter_time):
            if dic_r1[self.getTargetType()] < int(iter_time):
                print(0)
                dic_r1[self.getTargetType()] += 1
                self.clearRecords()
                steps,target_type = testMap.findTarget(by="belief"),self.getTargetType()
                if target_type == "Flat":
                    flat_r1.append(steps)
                elif target_type == "Hilly":
                    hill_r1.append(steps)
                elif target_type == "Forest":
                    forest_r1.append(steps)
                elif target_type == "Maze":
                    cave_r1.append(steps)
                self.resetTarget()
            else:
                self.resetTarget()
        print(1)
        while sum(dic_r2.values()) < 4 * int(iter_time):
            if dic_r2[self.getTargetType()] < int(iter_time):
                print(0)
                dic_r2[self.getTargetType()] += 1
                self.clearRecords()
                steps,target_type = testMap.findTarget(by="found"),self.getTargetType()
                if target_type == "Flat":
                    flat_r2.append(steps)
                elif target_type == "Hilly":
                    hill_r2.append(steps)
                elif target_type == "Forest":
                    forest_r2.append(steps)
                elif target_type == "Maze":
                    cave_r2.append(steps)
                self.resetTarget()    
            else:
                self.resetTarget()
        print(2)
        while sum(dic_u.values()) < 4 * int(iter_time):
            if dic_u[self.getTargetType()] < int(iter_time):
                print(0)
                dic_u[self.getTargetType()] += 1
                self.clearRecords()
                steps,target_type = testMap.findTarget(by="uniform"),self.getTargetType()
                if target_type == "Flat":
                    flat_u.append(steps)
                elif target_type == "Hilly":
                    hill_u.append(steps)
                elif target_type == "Forest":
                    forest_u.append(steps)
                elif target_type == "Maze":
                    cave_u.append(steps)
                self.resetTarget()
            else:
                self.resetTarget()
        
        print('FLAT','HILL','FOREST','CAVE')
        print('rule1:',np.mean(flat_r1),np.mean(hill_r1),np.mean(forest_r1),np.mean(cave_r1))
        print('rule2:',np.mean(flat_r2),np.mean(hill_r2),np.mean(forest_r2),np.mean(cave_r2))
        print('uniform:',np.mean(flat_u),np.mean(hill_u),np.mean(forest_u),np.mean(cave_u))
        print("")
        print('The total performance of rule1 is:',0.2*np.mean(flat_r1)+0.3*np.mean(hill_r1)+0.3*np.mean(forest_r1)+0.2*np.mean(cave_r1))
        print('The total performance of rule2 is:',0.2*np.mean(flat_r2)+0.3*np.mean(hill_r2)+0.3*np.mean(forest_r2)+0.2*np.mean(cave_r2))
        print('The total performance of uniform is:',0.2*np.mean(flat_u)+0.3*np.mean(hill_u)+0.3*np.mean(forest_u)+0.2*np.mean(cave_u))
        
testMap = Map()
#testMap.showMap()

#print(testMap.findTarget("belief"))
#print(testMap.findTarget("found"))


#result1,result2=testMap.testMethodByRule(loop=30)
#print("By rule 1, average search number for fixed map is",sum(result1)/30)
#print("By rule 2, average search number for fixed map is",sum(result2)/30)


#print(testMap.findTarget_return_type(by="uniform"))
testMap.test_rule1_rule2_uniform(10)

'''
result1,result2=testMap.testMethodWithMultiMap(loop=30)
print("By rule 1, average search number for multiple maps is",sum(result1)/30)
print("By rule 2, average search number for multiple maps is",sum(result2)/30)


#result1,result2=testMap.testMethodWithRestrict(loop=10,by="belief")
#print("If move to neighbor, average search number is",sum(result1)/10)
#print("If move by belief, average search number is",sum(result2)/10)        
        
                    
#result1,result2=testMap.testMethodWithRestrict(loop=10,by="found")
#print("If move to neighbor, average search number is",sum(result1)/10)
#print("If move by found probability, average search number is",sum(result2)/10)         

 
#result1,result2,result3=testMap.testMethodWithRestrictAmongThree(loop=10)
#print("If move to neighbor, average search number is",sum(result1)/10)
#print("If move by belief probability, average search number is",sum(result2)/10) 
#print("If move by found probability, average search number is",sum(result3)/10)


#plt.plot(result1,'-o',label='Move to neighbor')
#plt.plot(result2,'-o',label='Move by belief')
#plt.plot(result3,'-o',label='Move by found')
#plt.xlabel('Loop')
#plt.ylabel('Number of search')
#plt.title('Number of search for 10 tests')
#plt.legend(['Move to neighbor', 'Move by belief', 'Move by found'], 
#           loc='upper right')
#plt.show()


        
#print(testMap.findMovingTarget(by="belief"))
#print(testMap.findMovingTarget(by="found"))


#result1,result2=testMap.testMethodWithMovingTarget(loop=10)
#print("By rule 1, average search number for moving target is",sum(result1)/10)
#print("By rule 2, average search number for moving target is",sum(result2)/10)         
        

plt.plot(result1,'-o',label='Rule 1')
plt.plot(result2,'-o',label='Rule 2')
plt.xlabel('Loop')
plt.ylabel('Number of search')
plt.title('Number of search for 30 tests')
plt.legend(['Rule 1', 'Rule 2'], loc='upper right')
plt.show()
'''





