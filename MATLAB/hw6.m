%% HW6

% close all
% clear all
% clc
% 
% eps1 = 0.01;
% p = [eps1, 0, 1, -2];
% e1rs = roots(p);
% x1e1 = -sqrt(-1)/sqrt(eps1)-1-sqrt(eps1)*3/2*sqrt(-1)+4*eps1
% x2e1 =  sqrt(-1)/sqrt(eps1)-1+sqrt(eps1)*3/2*sqrt(-1)+4*eps1
% x3e1 = 2-8*eps1
% 
% eps2 = 0.0001; 
% p = [eps2, 0, 1, -2];
% e2rs = roots(p);
% x1e2 = -sqrt(-1)/sqrt(eps2)-1-sqrt(eps2)*3/2*sqrt(-1)+4*eps1
% x2e2 =  sqrt(-1)/sqrt(eps2)-1+sqrt(eps2)*3/2*sqrt(-1)+4*eps1
% x3e2 = 2-8*eps2

%% #3 on this aych dub

close all
clear all
clc

eps = 0.1; %eps
xmesh = linspace(0,1,50);
xsolved = linspace(0,1,1000);
ysolved = 1./(1+xsolved)+1-3/4.*exp(-xsolved./eps)-1; %eps
%plot(xsolved,ysolved)
hold on
solinit = bvpinit(xmesh,@guess);
sol = bvp5c(@bvpfcn,@bcfcn,solinit);
plot(solinit.x,solinit.y,'-o')
plot(sol.x,sol.y(1,1:end))
legend


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




