%% Homogenous System of ODEs

% this code is an example of solving a system of linear ODEs that have been
% broken down from an initial singular equation of more complicated ODEs

clear all; 
clc; 

syms y1(t) y2(t) y3(t); 

A = [-2 1 0; 0 -2 1; 0 0 -2]; 
B = [0;0;0]; 
Y = [y1;y2;y3]; 

odes = diff(Y) == A*Y + B

[y1Sol(t),y2Sol(t),y3Sol(t)] = dsolve(odes);
y1Sol(t) = simplify(y1Sol(t))
y2Sol(t) = simplify(y2Sol(t))
y3Sol(t) = simplify(y3Sol(t))
