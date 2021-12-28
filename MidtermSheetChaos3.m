%% Bif Diag. Real Case

clear all;
close all;
clc;
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
xlabel('{c}');
ylabel('{x_n}');
set(gca,'FontSize',30);
grid on;
axis([c_min c_max xmin xmax]);