%% DDE23 Solver Solution

% see Example of DDE23 #01 in files for explanation of code

clc;
clear all; 
close all; 

tau = 10;
tf = 300;
beta = 1.1;
gamma = 1; 
alpha = 100; 

sol1 = dde23(@ddefunc1,tau,@yhist,[-tau tf]);
% sol2 = dde23(@ddefunc2,lags,@yhist,[0 tf]);
% sol1 = dde23(@ddefunc,lags,@yhist,[0 tf]);
% sol1 = dde23(@ddefunc,lags,@yhist,[0 tf]);
% sol1 = dde23(@ddefunc,lags,@yhist,[0 tf]);

t = linspace(0,tf,20000);
y1 = deval(sol1, t);
% y2 = deval(sol2, t);

figure;
plot(t,y1);
hold on
yline((alpha/(gamma+beta)))
% plot(t,y2);
% legend
% hold off
    
function yp = ddefunc1(t,y,YL)
    yp = 100-1*y-1.1*YL;
end 

% function yp = ddefunc2(t,y,YL)
%     yp = 1*y*(1-YL);
% end 

function y = yhist(t)
    y = 150;
end