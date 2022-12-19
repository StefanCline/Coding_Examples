%% Final Project Code for 1D heat eq in varrying media

% Note to potential reader: 
% this code was generated under an extreme time crunch and therefore isn't 
% optimized. The paper outlining the details of this code can be found
% here: https://github.com/StefanCline/Coding_Examples/tree/main/Noteworthy_Projects
% 

close all
clear all
clc

%array of delta x values for cycling
view = 0; % change to 1 in order to see the plots
delx = [1/6, 1/12, 1/24];% , 1/48, 1/96];
delxcount = length(delx);
b = [3, 2, 1]; % b values from u_t = b*u_xx, in the paper this is D_i, b used initially as Strikwerda uses b
Bm = length(b);

% Heavy Content Load for Exact Solution
% view paper for details

%Lambda's computed in MAPLE, could be optimized by computing them in a
%single script in MATLAB, not done here due to time constraints

lambda = [1.397471655, 2.800269586, 4.009928200, 5.582827933, 6.847666687, 8.295778221, 9.558768567, 11.00922329, 12.44277444, 13.69468804];  
layers = 3;
LL = length(lambda);
CmNum = 0;
CmDen = 0;
% Diffusion coefficients
d1 = sqrt(3); d2 = sqrt(2); d3 = 1;
%Cm placeholder
Cm = zeros(1,LL);

%Note that the below are NOT optimized, these equations could be put into a
%loop to generalize the code and make it applicable for all robin BC
%problems

% Defining the X_im(x) functions
funX1 = @(x,Lam) (sin(Lam./d1.*(x)));
funX2 = @(x,Lam) (((d1./d2.*cos(Lam./d1).*sin(Lam./d2.*(x-1))) + (sin(Lam./d1).*cos(Lam./d2.*(x-1)))));
funX3 = @(x,Lam) (((d2./d3.*(((d1./d2.*cos(Lam./d1).*cos(Lam./d2))-(sin(Lam./d1)).*(sin(Lam./d2))))).*sin(Lam.*(x-2)))... 
        + (d1./d2.*cos(Lam./d1).*sin(Lam./d2)+sin(Lam./d1).*cos(Lam./d2)).*cos(Lam.*(x-2)));
%Squared X_im    
funX1sqr = @(x,Lam) (sin(Lam./d1.*(x))).^2;
funX2sqr = @(x,Lam) (((d1./d2.*cos(Lam./d1).*sin(Lam./d2.*(x-1))) + (sin(Lam./d1).*cos(Lam./d2.*(x-1))))).^2;
funX3sqr = @(x,Lam) (((d2./d3.*(((d1./d2.*cos(Lam./d1).*cos(Lam./d2))-(sin(Lam./d1)).*(sin(Lam./d2))))).*sin(Lam.*(x-2)))... 
        + (d1./d2.*cos(Lam./d1).*sin(Lam./d2)+sin(Lam./d1).*cos(Lam./d2)).*cos(Lam.*(x-2))).^2;
%Functions with g_i(x)
funX1g1 = @(x,Lam) (sin(Lam./d1.*(x))).*(4./11.*x);
funX2g2 = @(x,Lam) (((d1./d2.*cos(Lam./d1).*sin(Lam./d2.*(x-1))) + (sin(Lam./d1).*cos(Lam./d2.*(x-1))))).*(-2./11 + 6./11.*x);
funX3g3 = @(x,Lam) (((d2./d3.*(((d1./d2.*cos(Lam./d1).*cos(Lam./d2))-(sin(Lam./d1)).*(sin(Lam./d2))))).*sin(Lam.*(x-2)))... 
        + (d1./d2.*cos(Lam./d1).*sin(Lam./d2)+sin(Lam./d1).*cos(Lam./d2)).*cos(Lam.*(x-2))).*(-10./11.*x+30./11);

% Precomputing Coefficients C_m
for m = 1:LL
   CmNum = integral(@(x) funX1g1(x,lambda(m)),0,1);
   CmNum = integral(@(x) funX2g2(x,lambda(m)),1,2) + CmNum;
   CmNum = integral(@(x) funX3g3(x,lambda(m)),2,3) + CmNum;
   
   CmDen = integral(@(x) funX1sqr(x,lambda(m)),0,1);
   CmDen = integral(@(x) funX2sqr(x,lambda(m)),1,2) + CmDen;
   CmDen = integral(@(x) funX3sqr(x,lambda(m)),2,3) + CmDen;

   Cm(m) = CmNum / CmDen;
end


%Computing FDS alongside of Exact Solution

for q = 1: length(delx)
    
    x = 0:delx(q):3;
    %delt not optimized, hard set for this problem to make sure D_i mu
    %<=1/2 for largest D_i which in this case is 3
    delt = delx(q)^2/6;
    M = length(x);
    t = 0:delt:2.5;
    
    u_old = zeros(1,M);
    u_new = zeros(1,M);
    
    %Steady State Solutions, outside loop 
    x1 = 0:delx(q):1;
    x2 = 1:delx(q):2;
    x3 = 2:delx(q):3;
    y1 = -4/11*x1+2;
    y2 = -6/11*(x2-1)+18/11;
    y3 = -12/11*(x3-2)+12/11;
    %Used for exact solution
    MM = length(x1);
    v1 = zeros(1,MM); v2 = zeros(1,MM); v3 = zeros(1,MM);
    
    for T = t
        %below is for the IC
        if T == t(1) % establishing staggered start IC
           for X = 1:M
              if x(X) <= 1
                  u_old(X) = 2;
              elseif x(X) > 1 & x(X) <= 2
                  u_old(X) = 2;
              else
                  u_old(X) = -2*x(X)+6;
              end
           end
           u_new = u_old;
           u1 = 2 + x1.*0;
           u2 = 2 + x2.*0;
           u3 = 6 - 2*x3;      
           
        else
            % This portion is for the second time step and after
            % Below Portion for the FDS
            for X = 2:M-1
               u_new(1) = 2; u_new(end) = 0; % Hardcoding of static BCs
               if x(X) == 1 
                   % boundary layer one
               elseif x(X) == 2
                   % boundary layer two
               else
                   % typical intermediate points using 6.3.1 of Strikwerda
                   if x(X) < 1 
                       B = b(1);
                   elseif x(X) >1 & x(X) < 2
                       B = b(2);
                   else
                       B = b(3);
                   end
                   u_new(X) = delt*B*(u_old(X+1)-2*u_old(X)+u_old(X-1))/delx(q)^2 + u_old(X);
               end
            end
            for X = 1:M
               if x(X) == 1
                   LP1 = X;
               elseif x(X) == 2
                   LP2 = X;
               else 
                   
               end
            end
            % doing the averaging of the Boundary Layer points, x0 and x1
            u_new(LP1) = (b(2)*u_old(LP1+1)-(b(2)+b(1))*u_old(LP1)+b(1)*u_old(LP1-1))/(delx(q)^2)*delt+u_old(LP1);
            u_new(LP2) = (b(3)*u_old(LP2+1)-(b(3)+b(2))*u_old(LP2)+b(2)*u_old(LP2-1))/(delx(q)^2)*delt+u_old(LP2);
            
            % Exact Solution Computations
            for m = 1:LL
               % first for v1
               X1 = funX1(x1,lambda(m)).*Cm(m).*exp(-lambda(m).^2.*T);
               v1 = X1 + v1; 
               % for v2
               X2 = funX2(x2,lambda(m)).*Cm(m).*exp(-lambda(m).^2.*T);
               v2 = X2 + v2; 
               % for v3
               X3 = funX3(x3,lambda(m)).*Cm(m).*exp(-lambda(m).^2.*T);
               v3 = X3 + v3; 
            end
            u1 = -4/11.*x1+2 + v1;
            u2 = -6/11.*(x2-1)+18/11 + v2;
            u3 = -12/11.*(x3-2)+12/11 + v3;
            X1 = 0; X2 = 0; X3 = 0; v1 = zeros(1,MM); v2 = zeros(1,MM); v3 = zeros(1,MM);
            
          % Counter used to track progress during longer computational runs
%         end
%         if mod(T,100) == 0
%             disp(T)
%         end
        
        % Plotting the Exact, Numerical and Steady State solutions every
        % loop to be in a movie format
%         if T == t(1) | T == t(12) | T == t(24) | T == t(36) | T == t(48) | T == t(60) | T == t(72)...
%                      | T == t(84) | T == t(96) | T == t(109) | T == t(163) | T == t(217) | T == t(325) ...
%                      | T == t(end)
            if view == 1
                plot(x,u_new,'b-o','LineWidth',2)
                hold on
                plot(x1,y1,'-c','LineWidth',2)
                plot(x2,y2,'-c','LineWidth',2)
                plot(x3,y3,'-c','LineWidth',2)
                plot(x1,u1,'--r','LineWidth',2)
                plot(x2,u2,'--r','LineWidth',2)
                plot(x3,u3,'--r','LineWidth',2)
                ylim([0,2.5])
                title("Layered BCs, $t=$"+T,'Interpreter','latex','FontSize',16)
                xlabel("$x$, with $\Delta x=$"+delx(q),'Interpreter','latex','FontSize',16)
                ylabel('$u(x,t)$','Interpreter','latex','FontSize',16)
                xline(1)
                xline(2)
                xline(3)
                legend('Numerical','Steady State','','','Exact','Interpreter','latex','FontSize',14,'location','SouthWest')
                hold off 
                pause(.01)
            end
            
            %Taking images at various timesteps
%             if T == t(1) 
%                 saveas(gcf,'Exact_Approx_1.png')
%             elseif T == t(12) 
%                 saveas(gcf,'Exact_Approx_2.png')
%             elseif T == t(24) 
%                 saveas(gcf,'Exact_Approx_3.png')
%             elseif T == t(36)
%                 saveas(gcf,'Exact_Approx_4.png')
%             elseif T == t(48)
%                 saveas(gcf,'Exact_Approx_5.png')
%             elseif T == t(60)
%                 saveas(gcf,'Exact_Approx_6.png')
%             elseif T == t(72)
%                 saveas(gcf,'Exact_Approx_7.png')
%             elseif T == t(84)
%                 saveas(gcf,'Exact_Approx_8.png')
%             elseif T == t(96)
%                 saveas(gcf,'Exact_Approx_9.png')
%             elseif T == t(109)
%                 saveas(gcf,'Exact_Approx_10.png')
%             elseif T == t(163)
%                 saveas(gcf,'Exact_Approx_11.png')
%             elseif T == t(217)
%                 saveas(gcf,'Exact_Approx_12.png')
%             elseif T == t(325)
%                 saveas(gcf,'Exact_Approx_13.png')
%             elseif T == t(end)
%                 saveas(gcf,'Exact_Approx_14.png')
%             else
% 
%             end
%             pause(.01)
%         end
            
            %Critical Step of making sure the newest iteration becomes the
            %old one for the next time step
            u_old = u_new;
        end
    
        %Computing the Errors
        %first, putting the exact piecewise solution into one array
        ue = u1;
        ue(end) = [];
        ue = [ue,u2];
        ue(end) = [];
        ue = [ue,u3];

        % Sum for L2 Norm
        summeA = 0;
        for X = 1:M
            summeA = summeA + abs(u_new(X)-ue(X))^2;
        end
        %L2 Norm
        L2EA(q) = sqrt(delx(q)*summeA);
        %Sup Norm
        SupEA(q) = max(abs(u_new-ue));

        %Order of accuracy calculation for both L2 and Sup Norm
        if q > 1
            R2LA(q) = abs(log(L2EA(q)/L2EA(q-1)))/log(2);
            RSupA(q) = abs(log(L2EA(q)/L2EA(q-1)))/log(2);
        end

    %End of actions, loop restarts at next time step
    end
end

% error calculations requiring comparisons, hence the number of discrete
% time steps being cycled through has to be greater than one
if length(delx) > 1 
    Final_Error_Results = [["Delta x",delx]',["L2 Error",L2EA]',["Order r",R2LA]',["Inf Error",SupEA]',["Order r",RSupA]']
end