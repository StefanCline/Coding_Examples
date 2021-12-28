%HW 3 matlab assist

close all;
clear all; 
clc

% a = sym('a');
% b = sym('b');
% 
% A = [3 -36; 1 3];
% [v,d]=eig(A)
% inv(v)*A*v;
% 
% Q = [-1 0; 0 1];
% inv(Q)*A*Q

t = sym('t');
a = sym('a');
b = sym('b');
c = sym('c');

A = [a, 0, 0, a; 0 a b 0; 0 c a 0; a 0 0 a];
[v,d]=eig(A)

% J = [-1 1 0 0; 0 -1 0 0; 0 0 -1 1; 0 0 -1 -1];
% expm(J*t)
% 
% J2 = [exp(t), t*exp(t), 0; 0, exp(t), 0; 0, 0, exp(t)]; 
% 
% P = [1 1 0 0 -1; 0 1 1 0 1; 0 0 1 1 -1; -2 -6 -7 -3 1]; 
% rref(P)

% J1 = [exp(-t) 0 0 0; 0 exp(-t) 0 0; 0 0 exp(-t) 0; 0 0 0 exp(-t)]
% J2 = [1 t 0 0; 0 1 0 0; 0 0 cos(t) sin(t); 0 0 -sin(t) cos(t)]
% J1*J2
