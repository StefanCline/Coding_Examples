% Simple FEM Example for the 1D Laplace Equation

% Solving the equation: 
% -u''=f(x)     Note! f here is not an IC but a forcing function!
% as a steady state solution we've assumed the u_t=0 and hence are just 
% solving the Laplace equation i.e. the 1D heat Eq. steady state

% BCs: u(0)=0, u(1)=0 <-- double sink at boundaries, expect steady to 
% go to zero at the boundaries 

% Will eventually solve: c = A\b
%                      : u_h=sum(c_i*phi_i)

close all; clear all; clc;
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
lw = 1.5; % plot line widths
FS = 26;  % plot larger fonts
fs = 18;  % plot smaller fonts

% BC Types
uL = 0;  % u(L) = 
uR = 0;  % u(R) =
%uPL = 0; % u'(L) =
%uPR = 0; % u'(L) =

n = 6;                   % density of x axis, Note:# of line elements = n+1
L = 0;                   % left boundary in 1D
R = 1;                   % right boundary in 1D
x = linspace(L,R,(n+2)); % 1D domain
xe = linspace(L,R,1000); % Exact Solution domain
dx = x(2)-x(1);          % delta between evenly spaced domain values
f = ones(1,length(x));   % making the Forcing Function f(x)=1 
                         % NOTE! If you change this, the code must be
                         % changed drastically for the b_vals function!
f(1) = [];               % making the f vector size n
f(end) = [];             % making the f vector size n
f = f';                  % making into column vector
b_i = 0.*f;              % empty b_i matrix

% Need to solve the matrix equation in the form: 
% -->   u = inv(Aij)*b
% Setting up loops to calculate all of the values of bi and Aij
% A must be an nxn matrix, also note we don't need to solve for the BCs
% as they're given to us (again n+2 is the full size of the discrete
% domain)
A_ij = zeros(n,n);

for ii = 1:n % rows
    for jj = 1:n % columns
        A_ij(ii,jj) = A_vals(ii,jj,dx); %filling out our A matrix
    end
        b_i(ii) = b_vals(dx); %filling out the b column vector
end

c = A_ij\b_i; % using the \ instead of inv for efficiency
              % solving c gives us the c in 
              % u_h = sum(c_j*phi_j)

% Now, complete the approximation of the function u_h
% u_i = c_i*phi_i
% This is a bit tricky. Becuase we're dealing with discrete hat functions,
% if we simply look at each phi function, it's one at that value of x_i,
% but otherwise zero. Ex: c_3*phi_3 = [0,0,c_3,0...,0,0]. Then, summing
% them all together is adding 1xn matricies that are zero everywhere but at
% that index, so we simply get u = [c1,c2,c3,...,cn]
% Note if we don't use hat functions, this simple switch doesn't work. 
u = c;

u = [uL;u;uR]; % Adding our two BCs back to build the full u(x) for the full 
               % original x domain

% Plotting the final result and comparing to the exact
plot(x,u,'LineWidth',lw)
hold on
uexact = -1/2*xe.^2+1/2.*xe;
plot(xe,uexact,'LineWidth',lw)
title("FEM vs Exact Solution for: $-u''(x)=1$",'FontSize',FS)
xlabel("Domain for $x$: $\mathcal{D}=[0,1]$",'FontSize',FS)
ylabel("$u(x,\infty)$",'FontSize',FS)
legend('Approx','Exact','FontSize',fs)



%%
%%%%%%%%%%%%% Functions for Aij and bi %%%%%%%%%%%%%

function Aij = A_vals(ii,jj,dk)
    % note that A_ij = int^R_L (v'_i*v'_j) dx (v and phi here are the same)
    % 
    if ii == jj
        Aij = 2/dk;
    elseif abs(ii-jj) == 1
        if ii > jj 
            Aij = -1/dk;
        else % ii < jj
            Aij = -1/dk;
        end
    elseif abs(ii-jj)>=2
        Aij = 0;
    end    
end

function bi = b_vals(dx)
    % note that this is the simplest case
    % we set f(x)=1, so f*phi --> 1*phi --> int (phi) 
    % for all tophat functions int phi is just 1*dx = dx
    bi = dx;
end

