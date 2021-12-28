function ou

close all
clear all
clc

rng('shuffle')
tau = 1; 
xi0 = 1;
dt  = 0.01;
D   = 0.5;
N   = 5000;
T   = N*dt;

pd = makedist('Normal',0.01,sqrt(dt));
dW = random(pd);

xi    = zeros(1,N);     % preallocate for efficiency
xi(1) = xi0 - dt*xi0/tau + sqrt(2*D)*dW/tau;

for j=2:N
   dW    = random(pd);
   xi(j) = xi(j-1) - dt*xi(j-1)/tau + sqrt(2*D)*dW/tau;
end

plot([0:dt:T],[xi0,xi],'k-', 'LineWidth',3);
xlabel('t','FontSize',12)
ylabel('\eta(t)','FontSize',16,'Rotation',0,...
'HorizontalAlignment','right');
set(gca,'FontSize',40);
grid on;

hold on 

t = linspace(0,T,100);
y = 1 - exp(-t/10)*.6; 
plot(t,y,'LineWidth',3)
xline(0)
yline(0)
yline(1,'--k','LineWidth',3)