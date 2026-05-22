%************************************************************
% Written by Erfan Yazdandoost Hamedani, created on 14 November 2018.
%
% IMPORTANT: optimal solution is obtained using CVX to measure 
% the accuracy of the solution found 
%
% In this example, APDB algorithm is specified to solve Quadratic Constrained 
% Quadratic Programming (QCQP) 
%
% See section 5.1. of the corresponding paper https://arxiv.org/pdf/1803.01401.pdf
% for more details.
%
% APD_c corresponds to constant stepsizes (Theorem 2.1 part (I)). Resulting in a
% convergence rate of O(1/k).
% When the problem is strongly convex (mu>0) APD_sc uses non-constant stepsizes 
% (Theorem 2.1 part (II)). Resulting in a convergence rate of O(1/k^2).
%************************************************************
% min_{x} 0.5*x'*A*x+b'*x
% s.t.    0.5*x'*Q_i*x+d_i'*x-c_i<=0, for i=1:m,
%         -10<=x<=10
%************************************************************