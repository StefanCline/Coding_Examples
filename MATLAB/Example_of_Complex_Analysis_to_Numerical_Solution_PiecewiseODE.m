%% Numerically Solving for (note, could also be done using LAPALCE): 

% eps*y' + y = f(t/eps), y(0)=y0
% f(xi) is periodic with period T, i.e. f(xi+T)=f(xi) on [0,T]
% f(xi) = 1 when 0 <= xi < T/2
%       = 0 when T/2 <= xi < T



% The below is the iterative work used to then develop the general looping
% structure. Note that the equation was originally solved by first
% determining a solution to the ODE for the first half, then the second,
% then combining them at the 'hand off' point at T/2. To then find
% iterations beyond the first period it becomes messy and symbolic math
% fails us in terms of being friendly. This is when this algorithm is used
% below. The zig zag nature of the solution is the result of forcing
% continuity onto the system. 



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

