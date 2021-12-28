%HW 6 Problem 2 Simulation Using Delays

function Delay_Model_HW_6;

close all;
clear all;
clc;

% Parameters
alpha = 100;
beta = 1.1;
gamma = 1;

% Tau Value
tau = 10;

history = [20];
tspan   = [0 150];

% Solve the ODEs that arise when there is no delay.
sol1 = dde23(@ddefun,[],history,tspan,[],alpha,beta,gamma);

% Solve the DDEs that arise when there is a delay of tau.
sol2 = dde23(@ddefun, [tau], history, tspan,[], alpha,beta,gamma);

figure(1)
plot(sol1.x,sol1.y,'b-','Linewidth', 3), hold on;
plot(sol2.x,sol2.y,'k-','Linewidth', 3);
xlabel('Time t');
ylabel('x(t)');
legend('No Delay','Delay','Location','NorthEast');
set(gca,'FontSize',16);
grid on;

function dxdt = ddefun(t,x,Z,alpha,beta,gamma) % equation being solved
if isempty(Z)     % ODEs
   dxdt = alpha- beta*x(1) - gamma*x(1);
else
  xlag = Z(:,1);

  dxdt = alpha - beta*x(1) - gamma*xlag(1);
end