% Math Models 5 HW

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

% DF = [-k+lam*F-2*sig*S, -lam*S; -c*F, a-2*b*F-c*S]
% [v,d]=eig(DF)
