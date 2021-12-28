clear all;
close all; 
clc

alpha = [3.8317, 5.13562, 6.38016; 
    10.17347, 11.61984, 13.01520; 
    16.47063, 17.95982, 19.40942];

[x,y]  = meshgrid(-0.6:0.01:0.6);
    r  = sqrt(x.^2+y.^2);
theta  = atan2(y,x);
    n  = 5;
alphan = alpha(3,3);

B   = besselj(n,alphan*r);
Psi = B.*exp(n*i*theta);
U   = real(Psi);

tiledlayout(1,3)
nexttile
surf(x,y,U)
nexttile
imagesc(U)
nexttile
contour(x,y,U,10)

