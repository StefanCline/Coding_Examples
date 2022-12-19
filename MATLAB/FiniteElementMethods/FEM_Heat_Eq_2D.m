%% COOK THE 2D TURKEY
% 2D heat equation, with forcing
clear, clc, clf;

% Font size, for plotting
fs = 14;


%% Define the rectangular domain
% Discretize 2D rectangular domain into triangular elements
% and choose a basis of tent functions for interior nodes

Nx = 20; % number of x nodes
Ny = 20; % number of y nodes

% Inline function for creating linear elements
myelement =@(x,y,a,b) max(1 - abs(x-a) - abs(y-b),0);

[x,y] = meshgrid(1:Nx,1:Ny);
T = delaunay(x,y);

%Plotting some example elements
% z1 = myelement(x,y,6,5);
% z2 = myelement(x,y,10,5);
% 
% figure (1)
% trisurf(T, x, y, z1 + z2,'facecolor',[0.8500 0.3250 0.0980])
% alpha 0.75; hold on;
% trisurf(T,x,y,0.01+0*z2,'facecolor','w')
% alpha 0.75;
% axis equal; colormap jet;
% 
% title('Triangulation of Domain','Interpreter','latex','FontSize',fs)
% xlabel('$x$','Interpreter','latex','FontSize',fs)
% ylabel('$y$','Interpreter','latex','FontSize',fs)
% zlabel('$z$','Interpreter','latex','FontSize',fs)


%% Set up initial condition u0 = u(x,y,0)
% Speficy the values of u0 on the grid (xi,yj) = (i,j)

% Initial Gaussian
Amplitude = 10;
Damping = 0.01;
G =@(x,y,xo,yo) Amplitude*exp(-Damping*(x-xo).^2-Damping*(y-yo).^2);
%G =@(x,y,xo,yo) Amplitude*sin(10*pi*x) + Amplitude*sin(10*pi*y);

% Center the Gaussian on the grid
xo = Nx/2;
yo = Ny/2;

% Fill a matrix with the discretized initial conditions
u0 = zeros(Ny,Nx);
for j = 1:Ny
    for i = 1:Nx
        u0(j,i) = G(i,j,xo,yo);
    end
end

% Plot the continuous version of the IC alongside u0
figure (1)

% Continuous
subplot(2,1,1)
[X,Y] = meshgrid(linspace(1,Nx,5*Nx), linspace(1,Ny,5*Ny));
Z = G(X,Y,xo,yo);
surf(X,Y,Z)
colorbar
shading interp;
colormap hot;
axis equal
title('$u_0 = u(x,y,0)$ (Continuous)','Interpreter','latex','FontSize',fs)
xlabel('$x$','Interpreter','latex','FontSize',fs)
ylabel('$y$','Interpreter','latex','FontSize',fs)
zlabel('$z$','Interpreter','latex','FontSize',fs)

% Piece-wise linear approximation
subplot(2,1,2)
s = mesh(u0);
s.FaceColor = 'interp';
colorbar
shading interp;
colormap hot;
axis equal
title('$u_0 = u(x,y,0)$ (Piecewise linear)','Interpreter','latex','FontSize',fs)
xlabel('$x$','Interpreter','latex','FontSize',fs)
ylabel('$y$','Interpreter','latex','FontSize',fs)
zlabel('$z$','Interpreter','latex','FontSize',fs)


%% Time stepping algorithm
% FORWARD EULER: Stability conidition, dt <= 1/(2*((1/dx)^2+(1/dy)^2))
t0 = 0;  % initial time
tf = 5; % final time
dt = 0.01; % time discretization
Nt = ceil((tf-t0)/dt);

%Turk = questdlg('Cook the turkey?','God calling:','Oh yeah','Nay','Nay');

% Fill up the M matrix
% (Elements M_ij: phi(i)*phi(j) is zero unless adjacent/identical)

% Integral for tent functions overlapping diagonally
Mdiagonal = 1/12;
Kdiagonal = 0;

% Integral for left/right or up/down overlapping tent functions
Madjacent = 1/12;
Kadjacent = -20*Damping; % -k

% Integral for whole element squr
Msqur = 17/3 + 1/6;
Ksqur = 24*20*Damping; % 24k

% Create the M matrix
M = MakeM(Nx,Ny,Mdiagonal,Msqur);

% Create the K matrix
K = MakeK(Nx,Ny,Kadjacent,Ksqur);

% Define the forcing function
f = 0*ones(Nx*Ny,1);

% Wrap u0 into a vector and use it to initialize u
u0 = reshape(u0',[Nx*Ny,1]);
uold = u0;

figure (2)
for t = t0:dt:tf
    unew = uold + M\(dt*(f - K*uold));
    
    % Enforce periodic BC's
    unew(1:Ny) = 0;
    unew(end-Ny+1:end) = 0;
    uold = unew;

    disp(t)

    s = mesh(reshape(unew,[Nx,Ny]));
    s.FaceColor = 'interp';
    colorbar
    shading interp;
    colormap hot;
    axis equal
    title("$u = u(x,y,t)$, at $t =$ "+t,'Interpreter','latex','FontSize',fs)
    xlabel('$x$','Interpreter','latex','FontSize',fs)
    ylabel('$y$','Interpreter','latex','FontSize',fs)
    zlabel('$z$','Interpreter','latex','FontSize',fs)
    xlim([0 Nx])
    ylim([0 Ny])
    zlim([0 10])
    pause(dt)
end



%% Defining some helper functions for constructing matrices

function D = MakeD(Nx,squr,diagonal)
    D = squr*diag(ones(Nx,1)) + diagonal*diag(ones(Nx-1,1),1) + diagonal*diag(ones(Nx-1,1),-1);
end

% Define a function that returns the Iu and Id matrices
function Delta = MakeDelta(Ny,diagonal)
    Delta = diagonal*diag(ones(Ny,1)) + diagonal*diag(ones(Ny-1,1),1) + diagonal*diag(ones(Ny-1,1),-1);
end

% Define a function that places the DeltaU and DeltaD matrices in M
function Bingus = PlaceDelta(Nx,Ny,diagonal)
    Delta = MakeDelta(Ny,diagonal);
    
    % Create the upper portion
    DeltaBlockU = {0};
    for m = 1:Ny
        DeltaBlockU{m} = Delta;
    end
    DeltaBlockU = blkdiag(DeltaBlockU{1:end});
    DeltaBlockU = circshift(DeltaBlockU,Nx,2);

    % Create the lower portion
    DeltaBlockD = {0};
    for m = 1:Ny
        DeltaBlockD{m} = Delta;
    end
    DeltaBlockD = blkdiag(DeltaBlockD{1:end});
    DeltaBlockD = circshift(DeltaBlockD,Nx,2)';

    % Return the combined blocks
    Bingus = DeltaBlockU + DeltaBlockD;
end

% Define a function that returns the M matrix
function M = MakeM(Nx,Ny,diagonal,squr)
    
    % Add the D blocks that run down the diagonal
    Block = {0};
    for m = 1:Ny
        Block{m} = MakeD(Nx,squr,diagonal);
    end
    M = blkdiag(Block{1:end});

    % Add the Iu and Id blocks on the off-diagonals
    Bingus = PlaceDelta(Nx,Ny,diagonal);

    % Deliver the final spicy meatball
    M = M + Bingus;
end


function Delta = MakeKDelta(Ny,diagonal)
    Delta = diagonal*diag(ones(Ny,1));
end

% Define a function that places the DeltaU and DeltaD matrices in M
function Bingus = PlaceKDelta(Nx,Ny,diagonal)
    Delta = MakeKDelta(Ny,diagonal);
    
    % Create the upper portion
    DeltaBlockU = {0};
    for m = 1:Ny
        DeltaBlockU{m} = Delta;
    end
    DeltaBlockU = blkdiag(DeltaBlockU{1:end});
    DeltaBlockU = circshift(DeltaBlockU,Nx,2);

    % Create the lower portion
    DeltaBlockD = {0};
    for m = 1:Ny
        DeltaBlockD{m} = Delta;
    end
    DeltaBlockD = blkdiag(DeltaBlockD{1:end});
    DeltaBlockD = circshift(DeltaBlockD,Nx,2)';

    % Return the combined blocks
    Bingus = DeltaBlockU + DeltaBlockD;
end

% Define a function that returns the K matrix
function K = MakeK(Nx,Ny,diagonal,squr)
    
    % Add the D blocks that run down the diagonal
    Block = {0};
    for m = 1:Ny
        Block{m} = squr*diag(ones(Nx,1)) + diagonal*diag(ones(Nx-1,1),1) + diagonal*diag(ones(Nx-1,1),-1);
    end
    K = blkdiag(Block{1:end});

    % Add the Iu and Id blocks on the off-diagonals
    Bingus = PlaceKDelta(Nx,Ny,diagonal);

    % Deliver the final spicy meatball
    K = K + Bingus;
end