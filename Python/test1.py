## D-Field Generator
import matplotlib.pyplot as plt
import math
import numpy as np
## Naming variables to be used

t = -9
y = -9
i = 0
slope = 0
t1 = 0
y1 = 0
t2 = 0
y2 = 0

## User Input

print('')
print("The equation for 1st Order Linear DEs is in the form: y'+q(t)y=g(t)")
## Backend use: simply type in q_t and g_t
## q_t = t*y
## g_t = t
## q_t = (print(input("Enter the expression for q(t): ")))
## g_t = (print(input("Enter the expression for g(t): "))) 
print('')
print("Iter. of t | Slope, | Segment Points")

while i < 361:
    if t == -9:
        print("Batch y at: y=",y)
    if y < 10 and t < 10:
        # here do g(t)-q(t) to get y'
        q_t = 1/3*t*y
        g_t = .1*(t**2)
        display_q_t = str("1/3t")
        display_g_t = str(".1(t^2)")

        slope = g_t - q_t
        
        # plot the slope at that point as a line segment
        t1 = 10
        t2 = -10 
        y1 = slope*(t1-t)+y
        y2 = slope*(t2-t)+y

        theta1 = math.atan((y1-y)/(t1-t))
        
        t_pos = 1/4*math.cos(theta1)+t
        y_pos = 1/4*math.sin(theta1)+y
        t_neg = -1/4*math.cos(theta1)+t
        y_neg = -1/4*math.sin(theta1)+y
        
        print("t=",t," | ",format(slope,".2f"),"| t1,y1=(",(format(t_pos,".2f")),",",(format(y_pos,".2f")),")", " |  t2,y2=(",(format(t_neg,".2f")),",",(format(y_neg,".2f")),")")

        point1 = [t_pos,y_pos]
        point2 = [t_neg,y_neg]
        t_vals = [point1[0],point2[0]]
        y_vals = [point1[1],point2[1]]
        plt.plot(t_vals, y_vals, "k-")
        plt.grid()
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title("D-Field for 1st Order Linear D.E. Equation: y'+"+str(display_q_t)+"y="+str(display_g_t))
        
        t = t + 1
        i = i + 1
    if t == 10:
        t = -9
        y = y + 1

plt.show() 

print('')