## Euler's Method Approximation
## LV22(II)-Q1

## Problem: y'=1-y/x, y(2)=-1, h=0.5

import math

## First we'll utilize our initial conditions

step_h = 0.5
y_last = -1
x_last = 2
y_next = 0
x_next = 0
n = 0

print('')
print("n", " ", "x_n", "   ", "y_n")
print(n,',', format(x_last,".4f")+",", format(y_last,".4f"))

i = 0
while (i < 4):
    n = n + 1
    y_next = y_last + ((1 - y_last/x_last)*step_h)
    x_next = x_last + step_h
    print(n,',', format(x_next,".4f")+",",format(y_next,".4f"))
    y_last = y_next
    x_last = x_next
    i = i + 1
print('')

#End of Program