%% Roots of a Function with a Variable

% Here we're finding the roots of a cubic equation in the form of 
% ax^3+bx^2+cx+d =0 
% an adjustable parameter being epsilon, abbreviated eps1 or eps2, this
% then gives a numerical output. (This comes more specifically from
% perturbation analysis where we assume the variable y =
% y_0+eps*y1+eps^2*y_2...

close all
clear all
clc

eps1 = 0.01;
p = [eps1, 0, 1, -2];
e1rs = roots(p);
x1e1 = -sqrt(-1)/sqrt(eps1)-1-sqrt(eps1)*3/2*sqrt(-1)+4*eps1
x2e1 =  sqrt(-1)/sqrt(eps1)-1+sqrt(eps1)*3/2*sqrt(-1)+4*eps1
x3e1 = 2-8*eps1

eps2 = 0.0001; 
p = [eps2, 0, 1, -2];
e2rs = roots(p);
x1e2 = -sqrt(-1)/sqrt(eps2)-1-sqrt(eps2)*3/2*sqrt(-1)+4*eps1
x2e2 =  sqrt(-1)/sqrt(eps2)-1+sqrt(eps2)*3/2*sqrt(-1)+4*eps1
x3e2 = 2-8*eps2



