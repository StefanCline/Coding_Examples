%% Boundary Value Problem (BVP)

% Here a bvp has been solved using MATLABs solver and also done using
% epsilon perturbation analysis. The epsilon can be controlled allowing for
% different input values

close all
clear all
clc

eps = 0.01; %eps
xmesh = linspace(0,1,50);
xsolved = linspace(0,1,1000);
ysolved = 1./(1+xsolved)+1-3/4.*exp(-xsolved./eps)-1; %eps
plot(xsolved,ysolved)
hold on
solinit = bvpinit(xmesh,@guess);
sol = bvp5c(@bvpfcn,@bcfcn,solinit);
plot(solinit.x,solinit.y,'-g')
plot(sol.x,sol.y(1,1:end),'-r')
legend('$\mathcal{O}(\varepsilon)$ Approximation','BVP Solver','','Inner','interpreter','latex')


% Equations
function dydx = bvpfcn(x,y);
dydx = zeros(2,1);
dydx = [y(2)
       -(y(1))^2-(1/0.1)*y(2)]; %eps
end

%Boundary Conditions
function res = bcfcn(ya,yb);
res = [ya(1)-1/4
      yb(1)-1/2];
end

%initial guess
function g = guess(x);
g = [1./(1+x)+1-3/4.*exp(-x./.1)-1 %eps
     1/(x+1)];
end 


