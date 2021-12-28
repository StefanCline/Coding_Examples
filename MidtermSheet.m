clear all; 
close all;
clc

% a = sym('a');
% b = sym('b');
% c = sym('c');
% d = sym('d');
% e = sym('e');
% f = sym('f');
% g = sym('g');
% h = sym('h');
% j = sym('j');
% 
% A = [a b; c d]
% A2 = [a b c; d e f; g h j];
% A3 = [.1 .1 .1; .1 .1 .1; .1 .1 .1]
% I1 = eye(2);
% I2 = eye(3);
% % Output = (I-A)*((A-I)+(A^2-I)+(A^3-I)+(A^4-I)+(A^5-I)+(A^6-I)+(A^7-I)+(A^8-I)+(A^9-I)+(A^10-I)+(A^11-I)+(A^12-I))
% % norm(Output,1)
% % norm(Output,2)
% % norm(Output,inf)
% 
% inv(I1-A)
% inv(I2-A2)
% inv(I2-A3)

%% Number 1

% tspan = [0 .05];
% y0 = 0;
% [t,y] = ode45(@(t,y) (1-y)/(0.1), tspan, y0);
% plot(t,y,'b-')
% hold on
% y02 = -exp(-0.05/.1)+1;
% xline(.05)
% yline(-exp(-0.05/.1)+1)
% tspan2 = [0.05 .1];
% [t,y2] = ode45(@(t2,y2) -y2/.1, tspan2, y02);
% plot(t,y2,'r-')
% xline(.1)
% xline(0)
% yline(0)
% hold off

% Attempt with Actual Equations
% t1 =  linspace(.0, .05, 1000);
% t2 =  linspace(.05, .1, 1000);
% yp1  = -exp(-(t1/.1))+1;
% yp2  = (-exp(-(.1/2)/.1+1/2)+exp(1/2))*exp(-t2/.1);
% 
% t3 =  linspace(.1, .15, 1000);
% t4 =  linspace(.15, .2, 1000);
% yp3  = -exp(-((t3-.1*log(-yp2(end)+1)-.1)/.1))+1;
% yp4  = yp3(end)*exp(3*1/(2))*exp(-t4/.1);
% 
% t5 =  linspace(.2, .25, 1000);
% t6 =  linspace(.25, .3, 1000);
% yp5  = -exp(-((t5-.1*log(-yp4(end)+1)-(2*.1))/.1))+1;
% yp6  = yp5(end)*exp(5*1/(2))*exp(-t6/.1);
% 
% t7 =  linspace(.3, .35, 1000);
% t8 =  linspace(.35, .4, 1000);
% yp7  = -exp(-((t7-.1*log(-yp6(end)+1)-(3*.1))/.1))+1;
% yp8  = yp7(end)*exp(7*1/(2))*exp(-t8/.1);
% 
% t9 =  linspace(.4, .45, 1000);
% t10 = linspace(.45, .5, 1000);
% yp9  = -exp(-((t9-.1*log(-yp8(end)+1)-(4*.1))/.1))+1;
% yp10 = yp9(end)*exp(9*1/(2))*exp(-t10/.1);
% 
% plot(t1, yp1, 'b')
% hold on
% plot(t2, yp2, 'r')
% plot(t3, yp3, 'b')
% plot(t4, yp4, 'r')
% plot(t5, yp5, 'b')
% plot(t6, yp6, 'r')
% plot(t7, yp7, 'b')
% plot(t8, yp8, 'r')
% plot(t9, yp9, 'b')
% plot(t10,yp10,'r')
% xline(0)
% xline(.05,'--k')
% xline(.1,'--k')
% xline(.15,'--k')
% xline(.2,'--k')
% xline(.25,'--k')
% xline(.3,'--k')
% xline(.35,'--k')
% xline(.4,'--k')
% xline(.45,'--k')
% xline(.5,'--k')
% yline(0)
% hold off

n = 2;
endset = 18;

t1 =  linspace(.0, .05, 1000);
t2 =  linspace(.05, .1, 1000);
yp1  = -exp(-(t1/.1))+1;
yp2  = (-exp(-(.1/2)/.1+1/2)+exp(1/2))*exp(-t2/.1);
yp_first_last = yp1(end);
yp_second_last = yp2(end);
plot(t1, yp1, 'b')
hold on
plot(t2, yp2, 'b')
xline(0)
yline(0)
xline(.05,'--k')
xline(.1,'--k')
xlabel('t')
ylabel('y')

while n < endset
    t_first  = linspace(.1*(n-1), .1*(n-1)+.05, 1000);
    t_second = linspace(.1*(n-1)+.05, .1*n, 1000);
    yp_odd_current = -exp(-((t_first-.1*log(-yp_second_last+1)-((n-1)*.1))/.1))+1;
    yp_even_current  = yp_odd_current(end)*exp((2*(n-1)+1)*1/(2))*exp(-t_second/.1);
    yp_first_last = yp_odd_current(end);
    yp_second_last = yp_even_current(end);
    plot(t_first, yp_odd_current,'-b');
    plot(t_second, yp_even_current,'-b');
    xline(.05*n,'--k')
    xline(.1*n,'--k')
    n = n + 1;
end 



% s = sym('s');
% T = 17;
% F = -exp(-T*s)/(s);
% F2 = 1/s
% ilaplace(F)
% ilaplace(F2)

%% Problem 1
% eps = sym('eps');
% y_0 = sym('y_0');
% T = sym('T');
% syms s;
% F1 = 1/(eps*s^2+s);
% F2 = 1/(eps*s+1);
% F3 = 1/s^2*((1-exp(-s*(T/2)))/(1-exp(-s*T)))+y_0/s
% ilaplace(F1)
% ilaplace(F2)
% ilaplace(F3)


%% Problem 5
% 
% syms y1(t) y2(t)
% 
% A = [0 1; 3/(t^2), 1/t];
% B = [-16*t^2; 8*t]
% Y = [y1;y2];
% C = Y(1) == [4;-2]
% odes = diff(Y) == A*Y + B
% 
% [y1Sol(t),y2Sol(t)] = dsolve(odes,C)
