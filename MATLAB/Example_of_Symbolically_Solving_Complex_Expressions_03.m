%% Complex Expressions with variables example 3

% In this example we find the Jacobian of a matrix using symbolic math. We
% see for the more complex expressions this becomes relatively messy as
% outputs. 

clear all
clc

k = sym('k');
lam = sym('lam');
sig = sym('sig');
a = sym('a');
b = sym('b');
c = sym('c');
F = sym('F');
S = sym('S');
L = sym('L');

A = [a-L, 0; 0, -k-L];
det(A)

DF = [-k+lam*F-2*sig*S, -lam*S; -c*F, a-2*b*F-c*S]
[v,d]=eig(DF)
