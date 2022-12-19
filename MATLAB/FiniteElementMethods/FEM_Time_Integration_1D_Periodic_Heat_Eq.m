%% 1D Heat Equation (Periodic) 

close all; clear all; clc;
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');

lw = 1.5; % plot line widths
FS = 26;  % plot larger fonts
fs = 18;  % plot smaller fonts

% BC Types (picked Dirichlet BCs)
%uL = 1;  % u(L) = 
%uR = 1;  % u(R) =

k = 1;                   % constant from the PDE
n = 48;                  % density of x axis, Note:# of line elements = n+1
L = 0;                   % left boundary in 1D
R = 1;                   % right boundary in 1D
x = linspace(L,R,(n+2)); % 1D domain + BCs
x(end) = [];             % kill the last point as it's just the first point
                         % above line for periodic BCs
xs = linspace(L,R,1000); % Stead State Solution domain
dx = x(2)-x(1);          % delta between evenly spaced domain values
f = zeros(1,length(x));  % making the Forcing Function f(x)=0 to have easy convergence to 0 
                         % NOTE! If you change this, the code must be
                         % changed drastically for the b_vals function!
%f(1) = [];              % we want to keep the size of $f$ to be n+1 now
f(end) = [];             % periodic BCs
f = f';                  % making into column vector
ts = 0; tf = .035;       % start and end time points
Den = 400000;            % density of the time-steps
t = linspace(ts,tf,Den); % creating the time axis
dt = t(2)-t(1);          % delta t
view_max = 3;            % for nice plot limits
view_min = -3;           % for nice plot limits

% Building the Matricies that won't change every iteration
%      K and M as defined in the paper
% Note that our nice choice of f means that it's simply 
%      a comlumn vector filled with 0 (not alwasy the case)
M = zeros(n+1,n+1);
K = M;

for i = 1:(n+1) % n+1 for periodic BC
    for j = 1:(n+1) % n+1 for periodic BC
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
    f(i) = 0; %kept in case wanted to be adjusted, but zero for this ex
end
K(1,end) = -1/dx;
K(end,1) = -1/dx;
M(1,end) = dx/6;
M(end,1) = dx/6;

% The initial condition and steady state
% note that because we have fixed BCs, we'll solve n-2 points
% if we had Neumann BCs, we'd need to solve n points 
% i.e., be careful here!

u_old = 3*sin(4*pi*x); % IC of zeros at the boundaries, one elsewhere
u_new = zeros(1,(n+2));      % Empty matrix to hold the n+1 solution in time
usted = 0*xs; % plotting the steady state solution

% Completing the time integration 

% Plotting the initial condition
figure
plot(xs,usted,'--b','LineWidth',lw)                 % steady state solution
hold on  
grid on
plot([x,R],[u_old,(3*sin(4*pi*R))],'-r','LineWidth',lw)             % initial condition
xlabel('$x$','FontSize',FS)
ylabel('$u(x,t)$','FontSize',FS)
title("Periodic at $t=$"+0,'FontSize',FS)
ylim([(view_min+view_min/10),(view_max+view_max/10)])  % keeping the frame 'steady'
xlim([L-.1,R+.1])
legend('Steady State','Transient Sol.','location','northeast')
hold off
pause(.1)
Tstr = num2str(0);
filename = strcat("C:\Pictures\head1d_period\Heat_to_steady_t_",Tstr,".png");
%exportgraphics(gcf,filename)
count = 1;
timecount = ceil(length(t)/8);

% The actual integration 
for T = t(2:end) % this "T" acts like the 'nth' timestep in the paper
        u_new = M\(dt.*(f-k.*K*u_old')) + u_old'; % finding u_{n+1}
        count = count+1;
        if mod(count,timecount) == 0 || T == t(end)
            plot(xs,usted,'--b','LineWidth',lw)  % plotting the steady state
            hold on
            grid on
            plot([x,R],[u_new',u_new(1)],'-r','LineWidth',lw)
            xlabel('$x$','FontSize',FS)
            ylabel('$u(x,t)$','FontSize',FS)
            title("Periodic at $t=$"+T,'FontSize',FS)
            ylim([(view_min+view_min/10),(view_max+view_max/10)])  % keeping the frame 'steady'
            xlim([L-.1,R+.1])
            legend('Steady State','Transient Sol.','location','northeast')
            hold off
            pause(0.1)
            %Tstr = num2str(T);
            %filename = strcat("C:\Pictures\head1d_period\Heat_to_steady_t_",Tstr,".png");
            %exportgraphics(gcf,filename)
        end
        u_old = u_new'; % making the updated frame become the 'old' frame
end