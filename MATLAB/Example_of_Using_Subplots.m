%% Subplot example

close all
clear all
clc

leftbound = -.1;
rightbound = 1.1;

subplot(3,1,1)
eps = 3;
t = linspace(leftbound,rightbound,1000); 
yexact = (exp(t*sqrt(eps))-exp(-t*sqrt(eps)))/(exp(sqrt(eps))-exp(-sqrt(eps)));
yapprox = t + 1/6*eps*(t.^3 - t);
plot(t,yexact,'-b');
hold on
plot(t,yapprox,'--r')
xline(0)
yline(0)
xline(1,'--k')
title('\epsilon = 3','FontSize',16)

subplot(3,1,2)
eps = 1;
t = linspace(leftbound,rightbound,1000); 
yexact = (exp(t*sqrt(eps))-exp(-t*sqrt(eps)))/(exp(sqrt(eps))-exp(-sqrt(eps)));
yapprox = t + 1/6*eps*(t.^3 - t);
plot(t,yexact,'-b');
hold on
plot(t,yapprox,'--r')
xline(0)
yline(0)
xline(1,'--k')
title('\epsilon = 1','FontSize',16)

subplot(3,1,3)
eps = 0.1;
t = linspace(leftbound,rightbound,1000); 
yexact = (exp(t*sqrt(eps))-exp(-t*sqrt(eps)))/(exp(sqrt(eps))-exp(-sqrt(eps)));
yapprox = t + 1/6*eps*(t.^3 - t);
plot(t,yexact,'-b');
hold on
plot(t,yapprox,'--r')
xline(0)
yline(0)
xline(1,'--k')
title('\epsilon = 0.1','FontSize',16)


