function [rel_err_x,time_period_apd,iter_epoch]=...
    EGM(input,optsol,optval,reg_param,x0,y0,stop_criteria,max_iter)
%************************************************************
% IMPORTANT: optimal solution is required to measure the accuracy of the
% solution found
%
% The algorithm is designed to solve "Matrix Kernel Learning problem" with
% l2-norm soft margin
%
%************************************************************
% min_{x>=0, <b,x>=0} max_{y in simplex} L(x,y)
% where L(x,y)=-2<x,ones> + sum_{l=1}^M (c/r_l) y(l)*x^T*G(K_l^{tr})*x +
% lambda*norm(x)^2
%************************************************************
norm_optsol = 1+norm(optsol);abs_optval = 1+ abs(optval);
%---------- Unfolding Input ----------%
G_tr_class = input{1,1};
Kernel_tr_class = input{2,1};
Kernel_test_class = input{3,1};
G_tr_max_norm = input{4,1};
Ytrain = input{5,1};
Ytest = input{6,1};
trace_kernels = input{7,1};
%-------------------------------------%
epoch_counter = 0;
%--------------------------------------%
%------ Initialization ----------------%
rel_err_x = [];
m = length(x0);
y = y0;
x = x0;
x_old = x;
iter = 0;
time_apd = 0;
%--------------------------------------%
%------ parameter selection -----------%
c = sum(trace_kernels);
num_kern = length(trace_kernels);
coef = 0.5;         % this can be tuned
B = 15;         % a bound on x solution
L_yx = coef*2*num_kern*B*G_tr_max_norm;
L_xx = coef*(2*reg_param+6*G_tr_max_norm);
tau = 1/(L_xx+L_yx);

%---------- Main Algorithm -----------%
while iter<max_iter && norm(x-optsol)/norm_optsol > stop_criteria
    tic;
    iter = iter+1;
    % G2: gradient of L(x,y) with respect to y for x
    [Grad2, Grad1] = inner_prod_matvec(G_tr_class,x,true);
    % computing the gradient of L(x,y) with respect to x
    grad_x = -2*ones(m,1)+2*(Grad1)*((c./trace_kernels).*y)+2*reg_param*x;
    x_mid = x-tau*grad_x;x_mid = proj_pos_plane(x_mid,Ytrain);
    grad_y = ((c./trace_kernels).*Grad2);
    % project onto the simplex
    y_mid = projsplx(y+tau*grad_y);
    % the next two steps
    [Grad2, Grad1] = inner_prod_matvec(G_tr_class,x_mid,true);
    % computing the gradient of L(x,y) with respect to x
    grad_x = -2*ones(m,1)+2*(Grad1)*((c./trace_kernels).*y_mid)+2*reg_param*x_mid;
    x_new = x - tau * grad_x;x_new = proj_pos_plane(x_new,Ytrain);
    grad_y = ((c./trace_kernels).*Grad2);
    % project onto the simplex
    y_new = projsplx(y + tau*grad_y);

    x = x_new; y = y_new;

    epoch_counter = epoch_counter+1;
    time_apd = time_apd+toc;
    rel_err_x(epoch_counter,1) = norm(x_new-optsol)/norm_optsol;
    time_period_apd(epoch_counter,1) = time_apd;
    iter_epoch(epoch_counter,1) = iter;
end
tic;
epoch_counter = epoch_counter+1;
rel_err_x(epoch_counter,1) = norm(x_new-optsol)/norm_optsol;
time_apd = time_apd+toc;
time_period_apd(epoch_counter,1) = time_apd;
iter_epoch(epoch_counter,1) = iter;
fprintf(...
    'Iteration    Time    Rel. Error Solution\n');
fprintf('%d    %9.4f       %9.1e\n',iter,time_period_apd(epoch_counter),...
    rel_err_x(epoch_counter));
end