close all; clear all; clc;
FS = 28;
mu = 2.0;
FZ = @(z,fz) 1/sqrt(2*pi)*exp(fz^2/2)*(1/2)*(1/sqrt(2*pi)*exp(-(z-mu)^2/2) + 1/sqrt(2*pi)*exp(-(z+mu)^2/2));
[z,fz] = ode45(FZ,[-7.5,7.5],-0.2);
plot(z,fz,LineWidth=3)
grid
xlabel("$z$","FontSize",FS,'Interpreter','latex')
ylabel("$f(z)$","FontSize",FS,'Interpreter','latex')
title("Bijective Map $\mu=$"+mu+", $\sigma=$"+1,"FontSize",FS,'Interpreter','latex')