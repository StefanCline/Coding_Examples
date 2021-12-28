%% HW5 ODEs

close all
clear all
clc

leftbound = -.1;
rightbound = 1.1;

subplot(3,1,1)
eps = 3;
t = linspace(leftbound,rightbound,1000); 
yexact = (exp(t*sqrt(eps))-exp(-t*sqrt(eps)))/(exp(sqrt(eps))-exp(-sqrt(eps)));
yapprox = t + 1/6*eps*(t.^3 - t);
plot(t,yexact,'-b');
hold on
plot(t,yapprox,'--r')
xline(0)
yline(0)
xline(1,'--k')
title('\epsilon = 3','FontSize',16)

subplot(3,1,2)
eps = 1;
t = linspace(leftbound,rightbound,1000); 
yexact = (exp(t*sqrt(eps))-exp(-t*sqrt(eps)))/(exp(sqrt(eps))-exp(-sqrt(eps)));
yapprox = t + 1/6*eps*(t.^3 - t);
plot(t,yexact,'-b');
hold on
plot(t,yapprox,'--r')
xline(0)
yline(0)
xline(1,'--k')
title('\epsilon = 1','FontSize',16)

subplot(3,1,3)
eps = 0.1;
t = linspace(leftbound,rightbound,1000); 
yexact = (exp(t*sqrt(eps))-exp(-t*sqrt(eps)))/(exp(sqrt(eps))-exp(-sqrt(eps)));
yapprox = t + 1/6*eps*(t.^3 - t);
plot(t,yexact,'-b');
hold on
plot(t,yapprox,'--r')
xline(0)
yline(0)
xline(1,'--k')
title('\epsilon = 0.1','FontSize',16)

% eps_check = sym('eps_check')
% syms y(t)
% ode = diff(y,t,2) == eps_check*y;
% cond1 = y(0) == 0;
% cond2 = y(1) == 1;
% conds = [cond1, cond2];
% ySol(t) = dsolve(ode,conds)
% x = linspace(-.5,1.5,1000);
% subs(ySol(t),eps_check,eps); 
% ySol = ySol(t);
% plot(t,ySol)

% eps = 0.05;
% t = linspace(0,150,2000);
% yapprox = cos(t)-eps.*((t.*sin(t)./2));
% yexact  = cos(t.*sqrt(1+eps));
% 
% plot(t,yexact,'-b')
% hold on
% plot(t,yapprox,'--r')
% xline(0)
% yline(0)
% title('Approximation v Exact Solution')
% legend('Exact','Approx.','Location','northwest')
% ylabel('y','FontSize',14)
% xlabel('t','FontSize',14)
% hold off

%% 
close all
clear all
clc

subplot(2,1,1)
eps = .5;
a = 2;
t = linspace(0.00001,.99999,1000);
yapprox = a-a*t+eps^2*(a/6.*t.^3-a/2.*t.^2+a/3.*t);
yexact  = -a*cos(eps)*sin(eps.*t)/sin(eps)+a*cos(eps.*t);
plot(t,yexact,'-b')
hold on
plot(t,yapprox,'--r')
xline(0)
yline(0)
title('Approximation v Exact Solution: \epsilon = 0.5','FontSize',14)
legend('Exact','Approx.','Location','northeast','FontSize',12)
ylabel('y','FontSize',14)
xlabel('x','FontSize',14)
hold off

subplot(2,1,2)
eps = 1;
a = 2;
t = linspace(0.00001,.99999,1000);
yapprox = a-a*t+eps^2*(a/6.*t.^3-a/2.*t.^2+a/3.*t);
yexact  = (-a*cos(eps)*sin(eps.*t))/sin(eps)+a*cos(eps.*t);
plot(t,yexact,'-b')
hold on
plot(t,yapprox,'--r')
xline(0)
yline(0)
title('Approximation v Exact Solution: \epsilon = 1','FontSize',14)
legend('Exact','Approx.','Location','northeast','FontSize',12)
ylabel('y','FontSize',14)
xlabel('x','FontSize',14)
hold off

%% 4b taylor compare

clear all
close all
clc
tlast = 2.5;

subplot(3,1,1)
a = 2;
eps = 1/pi;
t = linspace(0,tlast,1000);
yexact = -a*cos(eps)*sin(eps.*t)./sin(eps)+a*cos(eps*t);
ytaylor = a - a*cos(eps)*eps.*t/sin(eps)-1/2*a*eps^2.*t.^2;
plot(t,yexact)
hold on
plot(t,ytaylor)
xline(0)
yline(0)
title('\epsilon = 1','FontSize',14)
hold off

subplot(3,1,2)
a = 2;
eps = 2/pi;
t = linspace(0,tlast,1000);
yexact = -a*cos(eps)*sin(eps.*t)./sin(eps)+a*cos(eps*t);
ytaylor = a - a*cos(eps)*eps.*t/sin(eps)-1/2*a*eps^2.*t.^2;
plot(t,yexact)
hold on
plot(t,ytaylor)
xline(0)
yline(0)
title('\epsilon = .5','FontSize',14)
hold off

subplot(3,1,3)
a = 2;
eps = 3/pi;
t = linspace(0,tlast,1000);
yexact = -a*cos(eps)*sin(eps.*t)./sin(eps)+a*cos(eps*t);
ytaylor = a - a*cos(eps)*eps.*t/sin(eps)-1/2*a*eps^2.*t.^2;
plot(t,yexact)
hold on
plot(t,ytaylor)
xline(0)
yline(0)
title('\epsilon = .1','FontSize',14)
hold off

%% ODE23 for #5
close all
clear all
clc

t0 = 0;
tf = 8; 
eps = 0.04;
tp = linspace(t0,tf,500);

syms u(t)
Du = diff(u);

ode = diff(u,t,2) == u + eps*t*u;
cond1 = u(0)  ==  1;
cond2 = Du(0) == -1;
conds = [cond1, cond2];
uSol(t) = dsolve(ode,conds);
uSol = simplify(uSol);
u_array = uSol(tp);
plot(tp,u_array,'-b','linewidth',2)
hold on
u_pert = exp(-tp)+eps.*(-exp(-tp)/8+exp(tp)./8-(tp.*(tp+1).*exp(-tp))./4);
plot(tp,u_pert,'-r','linewidth',2)
u_power = 1 - tp + 1/2.*tp.^2+((eps-1)/6).*tp.^3+((-2*eps+1)/24).*tp.^4+((4*eps-1)/120).*tp.^5+((4*eps^2-6*eps+1)/720).*tp.^6;
plot(tp,u_power,'-m','linewidth',2)
legend('Numerically Solved','Perturbation Method','Power Series','Location','northwest','FontSize',12)
title('Numerical, Perturbation and Taylor Solutions Comparison','FontSize',16)
xlabel('t','FontSize',14)
ylabel('u','FontSize',14)
ylim([0,10]);
grid on
hold off

%% prob 6 hw5

% close all
% clear all
% clc
% 
% syms y(t) t;
% e = 0.1;
% time = [0 50];
% TIME = linspace(0,50,1000);
% T = linspace(0,50,1000);
% % inish condish
% y0 = [1 0];
% y_approx = cos(T) + eps.*((sin(T).*(-cos(T).*sin(T)+T)./8));
% plot(T,y_approx,'--r','linewidth',1.5)
% hold on
% [t,y] = ode23(@(t,y) odefcn(t,y,e), time, y0);
% figure (1)
% plot(t,y(:,1),'-b','linewidth',1.5)
% title('Numerical and Perturbation Comparison','FontSize',16)
% grid on
% xlabel('t','FontSize',12)
% ylabel('y','FontSize',12)
% ylim([-1.1 1.1])
% legend('Numerical','Perturbation','FontSize',12)
% function dydt = odefcn(t,y,e)
% dydt = zeros(2,1);
% dydt(1) = y(2);
% dydt(2) = -y(1)+e*y(1)*(y(2))^2;
% end

% %% ODE23 for #7
% close all
% clear all
% clc
% 
% t0 = 0;
% tf = 8; 
% eps = 0.2;
% tp = linspace(t0,tf,500);
% 
% syms u(t) 
% Du = diff(u);
% 
% ode = diff(u,t,1) == -u + 1/(1+eps*u);
% cond1 = u(0)  ==  0;
% %cond2 = Du(0) == -1;
% conds = [cond1];
% uSol(t) = dsolve(ode,conds);
% uSol = simplify(uSol);
% u_array = uSol(tp);
% plot(tp,u_array,'-b','linewidth',2)
% hold on
% u_pert = 1-exp(-tp)+eps.*((tp-exp(tp)+1)*exp(-tp));
% plot(tp,u_pert,'-r','linewidth',2)
% % u_power = 1 - tp + 1/2.*tp.^2+((eps-1)/6).*tp.^3+((-2*eps+1)/24).*tp.^4+((4*eps-1)/120).*tp.^5+((4*eps^2-6*eps+1)/720).*tp.^6;
% % plot(tp,u_power,'-m','linewidth',2)
% legend('Numerically Solved','Perturbation Method','Power Series','Location','northwest','FontSize',12)
% title('Numerical, Perturbation and Taylor Solutions Comparison','FontSize',16)
% xlabel('t','FontSize',14)
% ylabel('u','FontSize',14)
% ylim([0,10]);
% grid on
% hold off

%% prob 7b hw5

close all
clear all
clc

syms y(t) t;
eps = 0.2;
tf = 10;
resolution = 500;
time = [0 tf];
TIME = linspace(0,tf,resolution);
T = linspace(0,tf,resolution);
% inish condish
y0 = [0];
y_approx = 1-exp(-T)+eps.*((T-exp(T)+1).*exp(-T));
plot(T,y_approx,'--r','linewidth',1)
hold on
[t,y] = ode23(@(t,y) odefcn(t,y,eps), time, y0);
figure (1)
plot(t,y(:,1),'-b','linewidth',1)
title('Numerical and Perturbation Comparison','FontSize',16)
grid on
xlabel('t','FontSize',12)
ylabel('u','FontSize',12)
%ylim([-1.1 1.1])
legend('Numerical','Perturbation','FontSize',12)
function dydt = odefcn(t,y,eps)
dydt = zeros(1,1);
dydt(1) = -y(1)+1/(1+eps*y(1));
% dydt(2) = -y(1)+e*y(1)*(y(2))^2;
end