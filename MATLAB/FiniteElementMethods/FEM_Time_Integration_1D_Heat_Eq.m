%% 1D Heat Equation 

close all; clear all; clc;
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');

lw = 1.5; % plot line widths
FS = 26;  % plot larger fonts
fs = 18;  % plot smaller fonts

% BC Types (picked Dirichlet BCs)
uL = 0;  % u(L) = 
uR = 0;  % u(R) =

k = 1;                   % constant from the PDE
n = 6;                   % density of x axis, Note:# of line elements = n+1
L = 0;                   % left boundary in 1D
R = 1;                   % right boundary in 1D
x = linspace(L,R,(n+2)); % 1D domain
xs = linspace(L,R,1000); % Stead State Solution domain
dx = x(2)-x(1);          % delta between evenly spaced domain values
f = ones(1,length(x));   % making the Forcing Function f(x)=1 
                         % NOTE! If you change this, the code must be
                         % changed drastically for the b_vals function!
f(1) = [];               % making the f vector size n
f(end) = [];             % making the f vector size n
f = f';                  % making into column vector
ts = 0; tf = .55;        % start and end time points
Den = 1000000;           % density of the time-steps
t = linspace(ts,tf,Den); % creating the time axis
dt = t(2)-t(1);          % delta t
Fmax = max(f);           % for nice plot limits

% Building the Matricies that won't change every iteration
%      K and M as defined in the paper
% Note that our nice choice of f means that it's simply 
%      a comlumn vector filled with del_x (not always the case)
M = zeros(n,n);
K = M;

for i = 1:n
    for j = 1:n
        if i == j
            M(i,j) = 2*dx/3;
            K(i,j) = 2/dx;
        elseif abs(i-j) == 1
            M(i,j) = dx/6;
            K(i,j) = -1/dx;
        else
            M(i,j) = 0;
            K(i,j) = 0;
        end
    end
    f(i) = dx;
end

% The initial condition and steady state
% note that because we have fixed BCs, we'll solve n-2 points
% if we had Neumann BCs, we'd need to solve n points 
% i.e., be careful here!

u_old = ones(1,n);          % IC of zeros at the boundaries, one elsewhere
u_new = zeros(1,n);         % Empty matrix to hold the n+1 solution in time
usted = -1/2*xs.^2+1/2.*xs; % plotting the steady state solution

% Completing the time integration 

% Plotting the initial condition
figure
plot(xs,usted,'--b','LineWidth',lw)      % steady state solution
hold on                                             
plot(x,[0,u_old,0],'-r','LineWidth',lw)  % initial condition
xlabel('$x$','FontSize',FS)
ylabel('$u(x,t)$','FontSize',FS)
title("1D Heat Eq. at $t=0$",'FontSize',FS)
ylim([0,1.25])
xlim([0,1])
legend('Steady State','Transient Sol.','location','northeast')
hold off
pause(.1)
count = 1;
timecount = ceil(length(t)/10);
% The actual integration 
for T = t(2:end) % this "T" acts like the 'nth' timestep in the paper
        u_new = M\(dt.*(f-k.*K*u_old')) + u_old'; % finding u_{n+1}
        count = count+1;
        if mod(count,timecount) == 0
            plot(xs,usted,'--b','LineWidth',lw) % plotting the steady state
            hold on
            plot(x,[0;u_new;0],'-r','LineWidth',lw)
            xlabel('$x$','FontSize',FS)
            ylabel('$u(x,t)$','FontSize',FS)
            title("1D Heat Eq. at $t=$"+T,'FontSize',FS)
            ylim([0,(Fmax+Fmax/10)])  % keeping the frame 'steady'
            xlim([0,1])
            legend('Steady State','Transient Sol.','location','northeast')
            hold off
            pause(0.1)
            Tstr = num2str(T);
            %filename = strcat("C:\Users\scline8247\Pictures\heat1d_dirch\Heat_to_steady_t_",Tstr,".png");
            %filename = convertCharsToStrings(filename);
            %exportgraphics(gcf,filename)
        end
        u_old = u_new'; % making the updated frame become the 'old' frame
end

