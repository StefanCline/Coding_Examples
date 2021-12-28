%% Example of Exact / Approx Comparison with LATEX labeling

clear all
close all
clc

vareps = 0.01

t = linspace(0,5,1000);
exact1 = 1./sqrt(exp(2.*t).*(1-vareps)+vareps);
% exact2 = -1./sqrt(exp(2.*t).*(1-vareps)+vareps);
approx = exp(-t)+.5.*(-exp(-3.*t)+exp(-t)).*vareps+exp(-t).*(exp(-3.*t)./2-3.*exp(-t)./2+1).*vareps^2;
plot(t,exact1,'-b')
hold on
% plot(t,exact2,'--b')
plot(t,approx,'-r')
xline(0)
yline(0)
xlabel('t','FontSize',16,'interpreter','latex')
ylabel('y','FontSize',16,'interpreter','latex')
title('Approximation vs Exact of $dy/dt + y = \epsilon y^3$','FontSize',16,'interpreter','latex')
hold off
