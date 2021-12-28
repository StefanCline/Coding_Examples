%% Midterm Chaos Matlab Sheet 2 for prob 3

close all; 
clear all;
clc

% theta1 = 0:0.001:pi;
% theta2 = pi:0.001:2*pi;
% c_1 = -1/4*((exp(i*theta1)-1).^2) + 1/4;
% polarplot(theta1,c_1)
% hold on
% c_2 = -1/4*((-exp(i*theta2)+1).^2) + 1/4;
% polarplot(theta2,c_2)
% hold off
% legend('c(\theta) 0 to \pi', 'c(\theta) \pi to 2\pi')
% 
% t = sym('t');
% c_1 = -1/4*((exp(i*t)-1).^2) + 1/4;
% c_2 = -1/4*((-exp(i*t)+1).^2) + 1/4;
% c_1-c_2

%% Stable Point Check

% last = 0;
% theta = 0;
% r = 0.4;
% for i = 1:300
%     Sec_last = last;
%     last = (last)^2 + -1/4*((r*exp(i*theta)-1).^2 + 1);
% end 
% last
% Sec_last

%% c=a+bi curves

% theta = linspace(0,2*pi,1000);
% a = (-((cos(theta)).^2)+2.*cos(theta)+(sin(theta)).^2)./4;
% b = (-2.*cos(theta).*sin(theta)+2.*sin(theta))./4;
% plot(a,b)
% hold on
% xline(0)
% yline(0)
% xlabel('Real values, a')
% ylabel('Imaginary values, bi')
% hold off

%% Random
% z1p = (1+sqrt(1-4*c))/2
% z1n = (1-sqrt(1-4*c))/2
% 
% z2p = (z1p)^2+c
% z2n = (z1n)^2+c


%% Roots
% c = sym('c');
% thet = sym('thet');
% p = [4, 0, 4, -exp(1i*thet)];
% rooteqs = roots(p);
% zeq1 = rooteqs(1);
% zeq2 = rooteqs(2);
% zeq3 = rooteqs(3);
% 
% theta = linspace(0,2*pi,1000);
% z1a  = subs(real(zeq1),thet,theta);
% z1bi = subs(imag(zeq1),thet,theta);
% z2a  = subs(real(zeq2),thet,theta);
% z2bi = subs(imag(zeq2),thet,theta);
% z3a  = subs(real(zeq3),thet,theta);
% z3bi = subs(imag(zeq3),thet,theta);
% 
% plot(z1a,z1bi)
% hold on
% plot(z2a,z2bi)
% plot(z3a,z2bi)
% xline(0)
% yline(0)
% xlabel('Real values, a')
% ylabel('Imaginary values, bi')
% hold off


%% Early Bif Diag. try
% cap_res = 5000;
% c_vals = linspace(-2.4,1.2,cap_res);
% X = 1;
% n = 1;
% x_next = 0;
% x_in = 0;
% xline(0)
% hold on
% yline(0);
% title('Bifurcation Diagram for, x_{n+1} = x_n + c, for varied values of c')
% xlabel('c','FontSize',16)
% ylabel('x_{n+1}','FontSize',16)
% 
% for X = 1:cap_res
%     for n = 1:50
%         x_next = (x_in)^2 + c_vals(X);
%         x_in = x_next;
%     end
%     if x_in < 3 & x_in > -3
%        plot(c_vals(X),x_in,'.r');
%     end
%     x_in = 0;
% end
% 
% hold off

%% f^(2)(z) c plots

% theta = linspace(0,2*pi,1000);
% a = (-((cos(theta)).^2)+2.*cos(theta)+(sin(theta)).^2)./4;
% b = (-2.*cos(theta).*sin(theta)+2.*sin(theta))./4;
% plot(a,b,'-b')
% hold on
% xline(0)
% yline(0)
% xlabel('Real values, a','FontSize',16)
% ylabel('Imaginary values, bi','FontSize',16)
% 
% theta = linspace(0,2*pi,3000); 
% a2 = 1/4.*cos(theta)-1;
% bi2 = 1/4.*sin(theta);
% a3 = -1/4.*cos(theta)-1
% bi3 = -1/4.*sin(theta)
% plot(a2,bi2,'--r')
% plot(a3,bi3,'--r')
% xline(0)
% yline(0)
% xlabel('Real values, a')
% ylabel('Imaginary values, bi')
% title('Combined Period 1 and 2 Stability Curves','FontSize',16)
% hold off
% % ylim([-.3 .3])

%% Sandbox

scatter(rand(1,10),rand(1,10))
cap_res = 200;
a_vals = linspace(-2.4,1.2,cap_res);
b_vals = linspace(-1.5,1.5,cap_res);
z_cur = 0+0*i;

plotmat = []
plotmat(2,2)=1
plotmat(200,200) = 7
