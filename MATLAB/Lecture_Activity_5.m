% Lecture_Activity_5

clear all
clc

A = [2 -6 4; -2 1 -2; 1 6 -1];

A_mat = A'*A
[v,d]=eig(A_mat)
sqrt(max(max(d)))
norm(A,2)

% % t = sym('t');
% [v,d]=eig(A); 
% P = [2 -1 -2; -1 0 1; -1 1 2]
% P_inv = inv(P);
% D = P_inv*A*P;

% t = [-1.5:.00001:1.2]; 
% y_1 = 2*exp(3*t);
% y_2 = -1*exp(-2*t);
% y_3 = exp(t);
% plot(y_3,y_2)
% hold on 
% xline(0)
% yline(0)
% hold off

% t = sym('t');
% D_sol = [exp(3*t), 0, 0; 0 exp(-2*t) 0; 0 0 exp(t)]
% P*D_sol*P_inv