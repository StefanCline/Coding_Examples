%% Solving Simple ODEs

clear all; 
close all; 
clc

tspan = [0 600];
Q0 = 15; 
[t,Q] = ode45(@(t,Q) -Q/100, tspan, Q0);
plot(t,Q,'-')
hold on
xline(20)

%% Problem 5

% M = 1000;
% c = 0.05;
% 
% tspan = [0 150];
% P0 = 100; 
% [t,P] = ode45(@(t,P) c*log(M/P)*P, tspan, P0);
% plot(t,P,'-')