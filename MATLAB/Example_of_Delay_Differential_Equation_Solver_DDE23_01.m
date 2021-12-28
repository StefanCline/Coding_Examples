%% Homework_6_Math_636

% This section shows how to use the delay differential equation functions
% by setting lags, we go back one unit of time
% tf is the ending point of the iterations
% each sol'N' = equation solves using the delay differential equation
% solver or dde23, and @yhist is from the deval(solN,t) where this solving
% of the equation becomes the historical required solving values of the dde

clc;
clear all; 
close all; 

lags = 1;
tf = 40;
sol1 = dde23(@ddefunc1,lags,@yhist,[0 tf]);
sol2 = dde23(@ddefunc2,lags,@yhist,[0 tf]);
sol3 = dde23(@ddefunc3,lags,@yhist,[0 tf]);

t = linspace(0,tf,5000);
y1 = deval(sol1, t);
y2 = deval(sol2, t);
y3 = deval(sol3, t);

figure;
plot(t,y1);
hold on
plot(t,y2);
plot(t,y3);
legend({'t_c = 2','t_c = 1','t_c = pi/2'},FontSize = 16)
hold off
    
function yp = ddefunc1(t,y,YL)
    yp = 2*y*(1-YL);
end 

function yp = ddefunc2(t,y,YL)
    yp = 1*y*(1-YL);
end 

function yp = ddefunc3(t,y,YL)
    yp = pi/2*y*(1-YL);
end 

function y = yhist(t)
    y = 3;
end