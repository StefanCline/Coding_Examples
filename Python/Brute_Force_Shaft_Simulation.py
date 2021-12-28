##Establishing a starting condition of a slight less than pi/2 offset

Theta_0 = 1.500796327
Ang_V_0 = 0
Ang_V_F = 0
Ang_Accl = 0
Theta_F = 0
Total_Time = 0
g = 9.80665
ss = 0.05
ss_squre = ss**2
Theta_Disp = 0
x = [Total_Time]
y = [Theta_0]

import math
import matplotlib.pyplot as plt

pie_div_2 = math.pi/2
print("Pi divided by 2 is:", pie_div_2)

i = 0
while i < 10000:
    i = i + 1
    cos = math.cos(Theta_0)
    Ang_Accl = g*cos
    Ang_V_F = Ang_V_0+Ang_Accl*ss
    Theta_F = Theta_0-(Ang_V_0*ss+0.5*Ang_Accl*ss_squre)
    Total_Time = Total_Time + ss
    if Ang_V_F < pie_div_2:
        Theta_Disp = abs(pie_div_2-Theta_F)
        Theta_F = Theta_F + 1.044*Theta_Disp
    else:
        Theta_Disp = abs(Theta_F - Theta_0)
        Theta_F = Theta_F - 1.044*Theta_Disp
    Theta_0 = Theta_F
    Ang_V_0 = Ang_V_F
    y.append(Theta_F)
    x.append(Total_Time)
    ## print(Theta_F, ", ", Total_Time)

print("The length of x: ", len(x))
print("The length of y: ", len(y))

plt.plot(x,y)
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('Theta over Time')
plt.show()