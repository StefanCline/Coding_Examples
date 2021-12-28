%% Symbolic Solving

close all
clear all
clc

syms x1(t) x2(t) x3(t)
A = [-2 0 0; 0 -1 1; 0 -1 -1];
B = [exp(-2*t); 1; t];
Y = [x1; x2; x3];
odes = diff(Y) == A*Y + B

% [x1Sol(t),x2Sol(t),x3Sol(t)] = dsolve(odes);
% x1Sol(t) = simplify(x1Sol(t))
% x2Sol(t) = simplify(x2Sol(t))
% x3Sol(t) = simplify(x3Sol(t))

x_10 = sym('x_10');
x_20 = sym('x_20');
x_30 = sym('x_30');

C = Y(0) == [x_10; x_20; x_30];
[x1Sol(t),x2Sol(t),x3Sol(t)] = dsolve(odes,C)


%%

syms x1(t) x2(t) 
s = sym('s');
A = [0 1; 2*s^(-2), -2*s^(-1)];
B = [6*s; 9*s^(-4)];
Y = [x1; x2];
odes = diff(Y) == A*Y + B

x_10 = sym('x_10');
x_20 = sym('x_20');

C = Y(1) == [x_10; x_20];
[x1Sol(t),x2Sol(t)] = dsolve(odes,C)








