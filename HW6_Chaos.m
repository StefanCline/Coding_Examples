%% HW_6_Chaos

close all;
clear all; 
clc

a = 3/4;

% while a <= 5/4
%     x = linspace(-2,2,100);
%     y = x.^4-2.*a.*x.^2+x+a^2-a;
%     plot(x,y);
%     hold on
%     a = a + 0.05;
% end
% yline(0)
% xline(0)
% y2 = x;
% plot(x,y2)
% legend
% hold off


% x = linspace(0,20,3000);
% a1 = 1./(4*x)+x.^2;
% plot(x,a1)
% hold on
% a2 = -1./(4*x)+x.^2;
% plot(x,a2)
% yline(0)
% yline(1)
% hold off

% b = sym('b')
% A = [1 0 -2*b 1 (b^2-b) 0; 0 -4 0 4*b -1 0; 0 -4 0 4*b 1 0]
% rref(A)

a = sym('a');
p = [1, 0, -2*a, 1, (a^2-a)];
r = roots(p)



