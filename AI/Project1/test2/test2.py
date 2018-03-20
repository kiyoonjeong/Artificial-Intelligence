from random import randint

def getTable(rows, columns):
    x = []
    for i in range(rows):
        x.append([])
        for j in range(columns):
            x[i].append(randint(1,5))
    return x

def neighbor(values, row, column):
    y = []
    if row != 0:
        y.append(values[row-1][column])
    if column != 0:
        y.append(values[row][column-1])
    if row != 0 and column != 0:
        y.append(values[row-1][column-1])
    if row != len(values)-1:
        y.append(values[row+1][column])
    if column != len(values[0])-1:
        y.append(values[row][column+1])
    if row != len(values)-1 and column != len(values[0])-1:
        y.append(values[row+1][column+1])
    if row != 0 and column != len(values[0])-1:
        y.append(values[row-1][column+1])
    if row != len(values)-1 and column != 0:
        y.append(values[row+1][column-1])
    maximum = max(y)
    count = len(y)
    print("Maximum is" , maximum, "and Count is", count)  
