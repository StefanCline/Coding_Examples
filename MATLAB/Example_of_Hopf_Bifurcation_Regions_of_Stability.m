%% Regions of Stability with Hopf bifurcations

clear all;
close all;
clc

xline(-1,'--b');
xlabel('a','FontSize',18);
ylabel('mu','FontSize',18);
title('(mu,a) plane','FontSize',18);
hold on 
a = linspace(-1.5,1.5,400);
mu1 = 2./(a.^2-1);
mu2 = -2./(a.^2-1);
plot(a,mu1,'--g');
plot(a,mu2,'--g');
yline(0,'--b');
xline(1,'--b');
xline(0,'-k');
hold off

a_mt  = [-2; -1.1; -0.9;  0; 1.1;  2; -2;   -1.1;  -0.9;   0;  1.1;   2];
mu_mt = [10;   .1;   .1; 10;  .1; 10; -10;   -.1;   -.1; -10;  -.1; -10];
n = 1;
while n < 13
    lam_1 = (-mu_mt(n)*(a_mt(n)^2-1)+sqrt((mu_mt(n)*a_mt(n)^2-mu_mt(n))^2-4))/2;
    disp('Lambda 1 for area')
    n
    disp(lam_1)
    lam_2 = (-mu_mt(n)*(a_mt(n)^2-1)-sqrt((mu_mt(n)*a_mt(n)^2-mu_mt(n))^2-4))/2;
    disp('Lambda 2 for area')
    n
    disp(lam_2)
    n = n + 1;
end 

%% Hopf Plot

omg_tau = linspace(-8.8,8.8);
mu1 = -omg_tau.*cot(omg_tau);
mu2 = omg_tau./(sin(omg_tau));
xline(0);
hold on
yline(0);
plot(mu1,mu2,'--b')
xlabel('\mu_1','FontSize',16);
ylabel('\mu_2','FontSize',16);
title('Hopf Curve','FontSize',20);
ylim([-14,14])
hold off

%% Part c for Q2

hold on
R = linspace(2,25,20001); 
alpha = [3.8317, 5.13562, 6.38016; 
    10.17347, 11.61984, 13.01520; 
    16.47063, 17.95982, 19.40942];

for c = 1:3
    for d = 1:3
        eps = 1 - 2.*((alpha(c,d)./R).^2) + ((alpha(c,d)./R).^4);
        plot(R,eps);
    end 
end 

ylabel('\epsilon','FontSize',16);
xlabel('R','FontSize',16);
yline(0);
xline(0);
ylim([0 1]);
legend('\epsilon_{1,1}', '\epsilon_{1,3}','\epsilon_{1,5}','\epsilon_{2,1}','\epsilon_{2,3}', '\epsilon_{2,5}', '\epsilon_{3,1}','\epsilon_{3,3}', '\epsilon_{3,5}', '', '','FontSize',14)
title('\epsilon_{nm} vs R','FontSize',16);

