%% WebWork_HW_Fundametal 

clear all; 
clc; 

syms y1(t) y2(t) y3(t); 

% ode1 = diff(x1) == 6*x1 - 3*x2;
% ode2 = diff(x2) == 4*x1 - x2;
% odes = [ode1; ode2]
% 
% S = dsolve(odes)
% 
% x1Sol(t) = S.x1
% x2Sol(t) = S.x2

A = [-2 1 0; 0 -2 1; 0 0 -2]; 
B = [0;0;0]; 
Y = [y1;y2;y3]; 

odes = diff(Y) == A*Y + B

[y1Sol(t),y2Sol(t),y3Sol(t)] = dsolve(odes);
y1Sol(t) = simplify(y1Sol(t))
y2Sol(t) = simplify(y2Sol(t))
y3Sol(t) = simplify(y3Sol(t))

% 
% A = [-2 1 0; 0 -2 1; 0 0 -2];
% A_2s = [-2 0 0; 0 -2 0; 0 0 -2]; 
% A_1s = [0 1 0; 0 0 1; 0 0 0]; 
% 
% disp('A_2 Matrix squared')
% A_2s^2
% disp('A_2 Matrix cubed')
% A_2s^3
% 
% disp('A_1 Matrix squared')
% A_1s^2
% disp('A_1 Matrix cubed')
% A_1s^3
% 
% syms ('t')
% 
% A_f = [1 t 1/2*t^2; 0 1 t; 0 0 1];
% B_f = [exp(-2*t) 0 0; 0 exp(-2*t) 0; 0 0 exp(-2*t)]; 
% 
% A_f*B_f
% 
% [v,d]=eig(A)
