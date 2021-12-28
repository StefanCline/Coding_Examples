%% ODE Numerical Solver for Order 2 epsilon perturbations


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

