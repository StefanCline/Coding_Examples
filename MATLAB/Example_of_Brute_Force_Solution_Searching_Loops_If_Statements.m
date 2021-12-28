%% Brute Force Example

% Here, a specific type of relationship is desired from two function types.
% Therefore, a brute force method is used to check for simple polynomial
% solutions between them. If solutions can be found, the answers are
% displayed. Also, trivial answers are ignored. 

close all;
clear all;
clc

%Note: G = 4x(1-x)
%Note: g = 2-x^2
%Note: C = ax+b
%Note: C_of_G = a(4x(1-x))+b
%Note: g_of_C = 2-(ax+b)^2

x = sym('x');

a = -20; 
b = -20; 
count = 1;

while a < 21 
    while b < 21
        C_of_G = a*(4*x*(1-x))+b;
        g_of_C = 2-(a*x+b)^2;
        gofC = subs(g_of_C,x,.7);
        CofG = subs(C_of_G,x,.7);
        if (gofC == CofG) & a ~=0
            a
            b
        end
        b = b + 1;
        count = count + 1;
        if mod(count,1000)==0
            count
        end
    end
        a = a + 1;
        b = -20; 
end

disp('Finished')