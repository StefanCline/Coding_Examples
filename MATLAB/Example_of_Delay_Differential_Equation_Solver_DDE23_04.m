%% DDE23 Solver Solution

% see Example of DDE23 #01 in files for explanation of code

clc;
clear all; 
close all; 

lags = 20;
tf = 300;
sol1 = dde23(@ddefunc1,lags,@yhist,[0 tf]);
% sol2 = dde23(@ddefunc2,lags,@yhist,[0 tf]);
% sol1 = dde23(@ddefunc,lags,@yhist,[0 tf]);
% sol1 = dde23(@ddefunc,lags,@yhist,[0 tf]);
% sol1 = dde23(@ddefunc,lags,@yhist,[0 tf]);

t = linspace(0,tf,5000);
y1 = deval(sol1, t);
% y2 = deval(sol2, t);

figure;
plot(t,y1);
% hold on
% plot(t,y2);
% legend
% hold off
    
function yp = ddefunc1(t,y,YL)
    yp = 40-0.3*y-0.1*YL;
end 

% function yp = ddefunc2(t,y,YL)
%     yp = 1*y*(1-YL);
% end 

function y = yhist(t)
    y = 150;
end