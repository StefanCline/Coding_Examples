%% Chaos HW9

close all;
clear all;
clc

Y = 1;
X = 1;
n = 1;
yline(0)
hold on
xline(0)
cap_res = 1000;
a_vals = linspace(-1.5,1.5,cap_res);
b_vals = linspace(-1.5,1.5,cap_res);
z_cur = a_vals(1)+b_vals(1)*i;
plotmat = [];
it_max = 20;

while Y < (cap_res+1)
    while X < (cap_res+1)
        z_next = ((a_vals(X) + b_vals(Y)*i));
        while n < it_max + 1
            z_cur = z_next .^ 2;
            % Check distance to be within 4
            dist = sqrt((real(z_cur)^2 + imag(z_cur)^2));
            if dist <= 10
                % black, actual contained
                plotmat(X,Y) = 300; 
                n = n + 1;
            elseif dist > 10 & mod(n,2) == 0
                % magenta, non-contained
                plotmat(X,Y) = 0;
                n = it_max + 1;
            else 
                % white, non-contained
                plotmat(X,Y) = 0;
                n = it_max + 1;
            end
            z_next = z_cur;
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
end
image(a_vals,b_vals,plotmat')
x = linspace(-1,1,100);
y1 =  sqrt(1-x.^2);
y2 = -sqrt(1-x.^2);
plot(x,y1,'--r','LineWidth',3)
plot(x,y2,'--r','LineWidth',3)