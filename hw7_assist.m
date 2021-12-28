%% Homework_7_Paper_Math_Models

clear all;
close all;
clc

% Parameters
k1 = .01;
k2 = .5;

x = linspace(-10,10,1000);
y = x';
z = k1-x-(4.*x.*y)/(1+x.^2);
surf(x,y,z)
% colormap(pink)
% shading interp
hold on



%Parameters
% k1 = .01;
% k2 = .5;
% x1 = 1;
% y1 = 10;
% x2 = 1;
% y2 = 1;
% 
% myfunction = @(x1,y1) (k1-x1-(4*x1*y1)/(1+x1^2));
% z1 = myfunction(x1,y1)
% c = 10;
% mygrid = @(x1,y1) ndgrid((-x1:x1/c:x1),(-y1:y1/c:y1));
% [x1,y1] = mygrid(0,2);
% z1 = k1-x1-(4*x1*y1)/(1+x1^2);
% mesh(x1,y1,z1)
% grid on
% hold on
% my_function = @(x2,y2) (k2*(x2-(x2*y2)/(1+x2^2)));
% z2 = my_function(x2,y2)
% c = 10;
% mygrid = @(x2,y2) ndgrid((-x2:x2/c:x2),(-y2:y2/c:y2));
% [x2,y2] = mygrid(-2,2);
% z2 = k2*(x2-(x2*y2)/(1+x2^2));
% mesh(x2,y2,z2)
% hold off