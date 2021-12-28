%% Bifurcation with plot

close all;
clear all;
clc;

% a = sym('a');
% x = sym('x');
% A = [(.7-sqrt(.49+4*a)), 0.3; 1, 0]; 
% [v,d]=eig(A);
% x = linspace(-1, 1, 3000);
% y1 = subs(d(1,1),a,x);
% plot(x,y1)
% hold on
% yline(-1)
% yline(1)
% hold off


% a = sym('a');
% x = sym('x');
% A = [(-0.7+sqrt(4*a-1.47)), 0.3; 1, 0]; 
% [v,d]=eig(A);
% x = linspace(-5, 5, 3000);
% d
% y1 = subs(d(2,2),a,x);
% plot(x,y1)
% hold on
% yline(-1)
% yline(1)
% hold off

% a = sym('a');
% x = sym('x');
% Df_1 = [(-0.7-sqrt(4*a-1.47)), 0.3; 1, 0];
% Df_2 = [(-0.7+sqrt(4*a-1.47)), 0.3; 1, 0];
% DF = Df_1*Df_2; 
% [v,d]=eig(DF);
% x = linspace(.912, .913, 3000);
% y1 = subs(d(1,1),a,x);
% plot(x,y1)
% hold on
% yline(-1)
% % yline(1)
% hold off

a = linspace(-.5,.5,1000);
x1 = (-0.7+sqrt((0.7^2)+4*a))/2
plot(a,x1)
title('x vs a')
xlabel('a axis')
ylabel('x axis')
hold on

x2 = (-0.7-sqrt((0.7^2)+4*a))/2
plot(a,x2)

xline(-0.1225)
xline(0.3675)
yline(.35)
yline(-0.35)

hold off







