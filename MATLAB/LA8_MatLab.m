%% LA_8 ODES & WebWork

close all;
clear all;
clc

% p = [1, 6, 12, 8];
% roots(p)

syms y(x);
Dy = diff(y);
ode = diff(y,x,2) == -8*x*y;

ySol(x) = dsolve(ode);
ySol = simplify(ySol)