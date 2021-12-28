%% Homework_7_Chaos

clear all;
close all;
clc;

% x = linspace(0,3.5,1000);
% y = -2*x.^2+8*x-5;
% plot(x,y)
% xlabel('x')
% ylabel('f(x)')
% title('Map of f(x)=-2x^2+8x-5')
% xline(0)
% yline(0)
% % hold on
% % y2 = x;
% % plot(x,y2)
% yline(1,'--')
% yline(2,'--')
% yline(3,'--')
% xline(1,'--')
% xline(2,'--')
% xline(3,'--')

%% ***************************************************

% x1 = 0;
% x2 = 1; 
% x3 = 2; 
% x4 = 3; 
% x5 = 4; 
% x6 = 5; 
% x7 = 6; 
% x8 = 7; 
% x9 = 8; 
% 
% plot(x1,x2,'r.')
% hold on
% plot(x2,x3,'r.')
% plot(x3,x4,'r.')
% plot(x4,x5,'r.')
% plot(x5,x6,'r.')
% plot(x6,x7,'r.')
% plot(x7,x8,'r.')
% plot(x8,x9,'r.')
% plot(x9,x1,'r.')
% plot([x8,x9],[x9,x1],'b')
% plot([x2,x8],[x3,x9],'b')
% plot([x1,0.25,0.75,x2],[x2,0,8,x3],'b')
% xline(0)
% yline(0)
% x = linspace(-1,9,100);
% y = x;
% plot(x,y,'m')
% grid
% hold off
 

%% ***************************************************

% x1 = 0;
% x2 = 1; 
% x3 = 2; 
% x4 = 3; 
% x5 = 4; 
% 
% plot(x1,x3,'r.')
% hold on
% plot(x2,x5,'r.')
% plot(x3,x4,'r.')
% plot(x4,x2,'r.')
% plot(x5,x1,'r.')
% plot([x1,x2,x3,x4,x5],[x3,x5,x4,x2,x1],'b')
% 
% xline(0)
% yline(0)
% x = linspace(-1,5,10);
% y = x;
% plot(x,y,'m')
% grid
% hold off

%% ***************************************************

x = linspace(0,1.2,1000);
% y1 = (2 + sqrt(5)).*x.*(2-x);
% y2 = (2+5).*x.*(1-x);
% y3 = 8.*x.*(1-x);
% y4 = 20.*x.*(1-x);

y5 = log(abs(2-4.*x+sqrt(5)-2*sqrt(5).*x+0.0000001-2.*x.*0.0000001));

% plot(x,y1)
% hold on
% plot(x,y2)
% plot(x,y3)
% plot(x,y4)
plot(x,y5)
hold on
% legend
xline(0)
yline(0)

grid
hold off