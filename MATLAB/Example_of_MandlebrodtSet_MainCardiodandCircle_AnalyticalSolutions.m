%% Analytical solving of largest mandelbrot images (main cardiod and first offset circle)

% The main cardiod but purely in polar, this curve shows the need for
% analysis in cartesian as this is meaningless as an image

close all; 
clear all;
clc

theta1 = 0:0.001:pi;
theta2 = pi:0.001:2*pi;
c_1 = -1/4*((exp(i*theta1)-1).^2) + 1/4;
polarplot(theta1,c_1)
hold on
c_2 = -1/4*((-exp(i*theta2)+1).^2) + 1/4;
polarplot(theta2,c_2)
hold off
legend('c(\theta) 0 to \pi', 'c(\theta) \pi to 2\pi')

t = sym('t');
c_1 = -1/4*((exp(i*t)-1).^2) + 1/4;
c_2 = -1/4*((-exp(i*t)+1).^2) + 1/4;
c_1-c_2

%% c=a+bi curves done in Cartesian, gives the main cardiod

theta = linspace(0,2*pi,1000);
a = (-((cos(theta)).^2)+2.*cos(theta)+(sin(theta)).^2)./4;
b = (-2.*cos(theta).*sin(theta)+2.*sin(theta))./4;
plot(a,b)
hold on
xline(0)
yline(0)
xlabel('Real values, a')
ylabel('Imaginary values, bi')
hold off

%% f^(2)(z) c plots, this gives the cirlce directly off of the main cardiod

theta = linspace(0,2*pi,1000);
a = (-((cos(theta)).^2)+2.*cos(theta)+(sin(theta)).^2)./4;
b = (-2.*cos(theta).*sin(theta)+2.*sin(theta))./4;
plot(a,b,'-b')
hold on
xline(0)
yline(0)
xlabel('Real values, a','FontSize',16)
ylabel('Imaginary values, bi','FontSize',16)

theta = linspace(0,2*pi,3000); 
a2 = 1/4.*cos(theta)-1;
bi2 = 1/4.*sin(theta);
a3 = -1/4.*cos(theta)-1
bi3 = -1/4.*sin(theta)
plot(a2,bi2,'--r')
plot(a3,bi3,'--r')
xline(0)
yline(0)
xlabel('Real values, a')
ylabel('Imaginary values, bi')
title('Combined Period 1 and 2 Stability Curves','FontSize',16)
hold off
% ylim([-.3 .3])
