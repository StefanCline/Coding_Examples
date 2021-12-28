%% Dsolve Example

close all;
clear all;
clc

syms y(x);
Dy = diff(y);
ode = diff(y,x,2) == -8*x*y;

ySol(x) = dsolve(ode);
ySol = simplify(ySol)