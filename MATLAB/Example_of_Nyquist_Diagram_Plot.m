%% Nyquist Plot with Tau Critical

close all;
clear all;
clc;

gamma = 1.1;
beta = 1;
tau = 10;
q2 = gamma;
p2 = [1 beta];
L2 = tf(q2,p2,'InputDelay',tau);

nyquist(L2);

h = findall(gcf, 'Type', 'Patch');  % Find all patch objects.

for i = 1:length(h)         % Loop through every patch object handle.
    h(i).LineWidth = 5;     % Set the new LineWidth value.
end

set(gca,'FontSize',16);
axis([-2 2], [-2 2]);
%grid on;
% 
% a1 = -2;
% a2 = 1;
% alpha = a1*a2;
% % tau = acos((alpha+2)/alpha)/(sqrt(-1-alpha));
% tau = 2
% 
% qlam = -alpha;
% plam = [1,2,1];
% L2 = tf(qlam,plam,'InputDelay',tau);
% 
% nyquist(L2);
% 
% h = findall(gcf, 'Type', 'Patch');  % Find all patch objects.
% 
% for i = 1:length(h)         % Loop through every patch object handle.
%     h(i).LineWidth = 5;     % Set the new LineWidth value.
% end
% 
% set(gca,'FontSize',16);
% axis([-2 2], [-2 2]);
%grid on;