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
title('$\varepsilon = 1$','FontSize',14,'interpreter','latex')
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
title('$\varepsilon = 0.5$','FontSize',14,'interpreter','latex')
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
title('$\varepsilon = .1$','FontSize',14,'interpreter','latex')
hold off

