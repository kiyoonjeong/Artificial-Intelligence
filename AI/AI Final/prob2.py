# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:34:41 2017

@author: Kiyoon Jeong
"""
#2-(a)
n = 100
u1 = 90
u2 = 80
u3 = 70
u4 = 60
u5 = 50
u6 = 40
u7 = 30
u8 = 20
d = -250

check_iter = 10

while abs(check_iter) > 0.00001:
    check_iter = n - max(100 + 0.9*u1 , -250 + 0.9*n)
    n = max(100 + 0.9*u1 , -250 + 0.9*n)
    u1 = max(90 + 0.9 *( 0.9 * u1 + 0.1 * u2) , -250 + 0.9*n)
    u2 = max(80 + 0.9 *( 0.8 * u2 + 0.2 * u3) , -250 + 0.9*n)
    u3 = max(70 + 0.9 *( 0.7 * u3 + 0.3 * u4) , -250 + 0.9*n)
    u4 = max(60 + 0.9 *( 0.6 * u4 + 0.4 * u5) , -250 + 0.9*n)
    u5 = max(50 + 0.9 *( 0.5 * u5 + 0.5 * u6) , -250 + 0.9*n)
    u6 = max(40 + 0.9 *( 0.4 * u6 + 0.6 * u7) , -250 + 0.9*n)
    u7 = max(30 + 0.9 *( 0.3 * u7 + 0.7 * u8) , -250 + 0.9*n)
    u8 = max(20 + 0.9 *( 0.2 * u8 + 0.8 * d) , -250 + 0.9*n)
    d = -250 + 0.9 * n
 
print(n, u1, u2, u3, u4, u5, u6, u7, u8, d)

#2-3


check_p = 250
minimax = [0,250]
p = 125

while abs(check_p - p) > 0.001:
    n = 100
    u1 = 90
    u2 = 80
    u3 = 70
    u4 = 60
    u5 = 50
    u6 = 40
    u7 = 30
    u8 = 20
    d = -250
    
    un = 100
    uu1 = 90
    uu2 = 80
    uu3 = 70
    uu4 = 60
    uu5 = 50
    uu6 = 40
    uu7 = 30
    uu8 = 20
    ud = -250
        
    check_iter = 10
    while abs(check_iter) > 0.00001:
        check_iter = n - max(100 + 0.9*u1 , -250 + 0.9*n)
        n = max(100 + 0.9*u1 , -250 + 0.9*n)
        u1 = max(90 + 0.9 *( 0.9 * u1 + 0.1 * u2) , -250 + 0.9*n)
        u2 = max(80 + 0.9 *( 0.8 * u2 + 0.2 * u3) , -250 + 0.9*n)
        u3 = max(70 + 0.9 *( 0.7 * u3 + 0.3 * u4) , -250 + 0.9*n)
        u4 = max(60 + 0.9 *( 0.6 * u4 + 0.4 * u5) , -250 + 0.9*n)
        u5 = max(50 + 0.9 *( 0.5 * u5 + 0.5 * u6) , -250 + 0.9*n)
        u6 = max(40 + 0.9 *( 0.4 * u6 + 0.6 * u7) , -250 + 0.9*n)
        u7 = max(30 + 0.9 *( 0.3 * u7 + 0.7 * u8) , -250 + 0.9*n)
        u8 = max(20 + 0.9 *( 0.2 * u8 + 0.8 * d) , -250 + 0.9*n)
        d = -250 + 0.9 * n
        
        un = max(100 + 0.9*uu1 , -p + 0.9*(0.5*uu1 + 0.5*uu2))
        uu1 = max(90 + 0.9 *( 0.9 * uu1 + 0.1 * uu2) , -p + 0.9*(0.5*uu1 + 0.5*uu2))
        uu2 = max(80 + 0.9 *( 0.8 * uu2 + 0.2 * uu3) , -p + 0.9*(0.5*uu1 + 0.5*uu2))
        uu3 = max(70 + 0.9 *( 0.7 * uu3 + 0.3 * uu4) , -p + 0.9*(0.5*uu1 + 0.5*uu2))
        uu4 = max(60 + 0.9 *( 0.6 * uu4 + 0.4 * uu5) , -p + 0.9*(0.5*uu1 + 0.5*uu2))
        uu5 = max(50 + 0.9 *( 0.5 * uu5 + 0.5 * uu6) , -p + 0.9*(0.5*uu1 + 0.5*uu2))
        uu6 = max(40 + 0.9 *( 0.4 * uu6 + 0.6 * uu7) , -p + 0.9*(0.5*uu1 + 0.5*uu2))
        uu7 = max(30 + 0.9 *( 0.3 * uu7 + 0.7 * uu8) , -p + 0.9*(0.5*uu1 + 0.5*uu2))
        uu8 = max(20 + 0.9 *( 0.2 * uu8 + 0.8 * ud) , -p + 0.9*(0.5*uu1 + 0.5*uu2))
        ud = -250 + 0.9 * (0.5*uu1 + 0.5*uu2)
        
    if u1 < uu1:
        minimax[0] = p
        check_p = p
        p = (minimax[1]+minimax[0])/2

    else:
        minimax[1] = p
        check_p = p
        p = (minimax[1]+minimax[0])/2

print(p)


#2-4


choice = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999, 0.9999]
for i in range(len(choice)):
    beta = choice[i]
    n = 100
    u1 = 90
    u2 = 80
    u3 = 70
    u4 = 60
    u5 = 50
    u6 = 40
    u7 = 30
    u8 = 20
    d = -250
    
    check_iter = 10
    while abs(check_iter) > 0.00001:
        check_iter = n - max(100 + beta*u1, -250 + beta*n)
        n = max(100 + beta*u1 , -250 + beta*n)
        u1 = max(90 + beta*( 0.9 * u1 + 0.1 * u2) , -250 + beta*n)
        u2 = max(80 + beta*( 0.8 * u2 + 0.2 * u3) , -250 + beta*n)
        u3 = max(70 + beta*( 0.7 * u3 + 0.3 * u4) , -250 + beta*n)
        u4 = max(60 + beta*( 0.6 * u4 + 0.4 * u5) , -250 + beta*n)
        u5 = max(50 + beta*( 0.5 * u5 + 0.5 * u6) , -250 + beta*n)
        u6 = max(40 + beta*( 0.4 * u6 + 0.6 * u7) , -250 + beta*n)
        u7 = max(30 + beta*( 0.3 * u7 + 0.7 * u8) , -250 + beta*n)
        u8 = max(20 + beta*( 0.2 * u8 + 0.8 * d) , -250 + beta*n)
        d = -250 + beta* n
        
    print(n,u1,u2,u3,u4,u5,u6,u7,u8,d)
    
#when the beta is over 0.99, it print out the optimum status. From u4, it should be replaced.


check_beta = 1
minimax = [0.9, 0.99]
beta = 0.95

while check_beta > 0.000001:   
    n = 100
    u1 = 90
    u2 = 80
    u3 = 70
    u4 = 60
    u5 = 50
    u6 = 40
    u7 = 30
    u8 = 20
    d = -250
    check_iter = 10

    while abs(check_iter) > 0.00001:
        check_iter = n - max(100 + beta*u1 , -250 + beta*n)
        n = max(100 + beta*u1 , -250 + beta*n)
        u1 = max(90 + beta*( 0.9 * u1 + 0.1 * u2) , -250 + beta*n)
        u2 = max(80 + beta*( 0.8 * u2 + 0.2 * u3) , -250 + beta*n)
        u3 = max(70 + beta*( 0.7 * u3 + 0.3 * u4) , -250 + beta*n)
        u4 = max(60 + beta*( 0.6 * u4 + 0.4 * u5) , -250 + beta*n)
        u5 = max(50 + beta*( 0.5 * u5 + 0.5 * u6) , -250 + beta*n)
        u6 = max(40 + beta*( 0.4 * u6 + 0.6 * u7) , -250 + beta*n)
        u7 = max(30 + beta*( 0.3 * u7 + 0.7 * u8) , -250 + beta*n)
        u8 = max(20 + beta*( 0.2 * u8 + 0.8 * d) , -250 + beta*n)
        d = -250 + beta* n
            
    if u4 != u5:
        minimax[0] = beta
        check_beta = beta
        beta = (minimax[1]+minimax[0])/2
        check_beta = abs(check_beta - beta)

    else:
        minimax[1] = beta
        check_beta = beta
        beta = (minimax[1]+minimax[0])/2
        check_beta = abs(check_beta - beta)


print(beta)


     