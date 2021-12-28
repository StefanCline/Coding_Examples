%% Chaos Midterm Sheet

close all;
clear all;
clc

Y = 1;
X = 1;
n = 1;
yline(0)
hold on
xline(0)
cap_res = 5000;
a_vals = linspace(-1.5,2.4,cap_res);
b_vals = linspace(-2,2,cap_res);
z_cur = 0+0*i;
plotmat = [];
it_max = 100;

while Y < (cap_res+1)
    while X < (cap_res+1)
        while n < it_max + 1
            z_next = (z_cur)^2 - a_vals(X) - b_vals(Y)*i;
            z_cur = z_next;
            % Check distance to be within 4
            dist = sqrt((real(z_cur)^2+imag(z_cur)^2));
            if dist <= 4
                % black, actual contained
                plotmat(X,Y) = 170; 
                n = n + 1;
            elseif dist > 4 & mod(n,2) == 0
                % magenta, non-contained
                plotmat(X,Y) = 0;
                n = it_max + 1;
            else 
                % white, non-contained
                plotmat(X,Y) = 0;
                n = it_max + 1;
            end
        end 
        X = X + 1;
        if X < cap_res+1
            n = 1;
        end 
        z_cur = 0;
    end 
    Y = Y + 1;
    if Y < cap_res+1
        X = 1;
        n = 1;
    end
%     cap_res - Y
end
image(a_vals,b_vals,plotmat')

Niterates   = 700;
Nc     = 1000;
Ntransients = 500;
%  ---------  Initial Conditions  ---------
c_min = -2.4;
c_max = 1.2;
xmin       = -1.0;
xmax       = 1.0;
for k=1:Nc
  c = c_min + (c_max-c_min)*(k-1)/(Nc-1);
  %  ---------  Transients  ---------
  x0 = 0.1237;
  for i=1:Ntransients
    x1 = x0.^2 + c;
    x0 = x1;
  end;
  
  %  ---------  Iterate  ---------
  for j=1:Niterates
    x1     = x0.^2 + c;
    x0     = x1;
    t(j,k) = c;
    v(j,k) = x1;
  end;
end;
plot(t,v,'r.','Markersize',4);
xlabel('{a}');
ylabel('{b}');
title('Combined Real Bifurcation and Mandelbrot')
set(gca,'FontSize',18);
grid on;

hold off;
