function [rel_err_x,time_period_apd,iter_epoch]=...
    rAPDB_yx_ada(input,optsol,optval,reg_param,x0,y0,stop_criteria,max_iter,K,non_mono)
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
epoch_counter = 0; mu=2;
%------ Initialization ----------------%
rel_err_x = [];
m = length(x0); n = length(y0);
y = y0;x = x0; x_old = x; y_old = y;
iter = 0; time_apd = 0; inner = zeros(max_iter,1);
%--------------------------------------%
%------ parameter selection -----------%
xi = 0.04;
c = sum(trace_kernels);
num_kern = length(trace_kernels);
coef = 0.5;         % this can be tuned
B = 15;         % a bound on x solution
L_yx = coef*2*num_kern*B*G_tr_max_norm; L_xx = coef*(2*reg_param+6*G_tr_max_norm);
tau = 1/(L_xx+L_yx)*10; gamma = 1/(L_yx)/tau;
eta = 0.7;
c_alpha = 0.4; delta = 0.5;
sigma_old = gamma*tau;
tau_old = tau;
alpha_old = c_alpha /sigma_old;
x_cum = zeros(m,1);y_cum = zeros(n,1); sigma_0 = sigma_old; TK = 0;
%---------- Main Algorithm -----------%
while iter < max_iter && norm(x-optsol)/norm_optsol > stop_criteria
    iter = iter+1; tic;
    while true
        sigma = gamma*tau;
        theta = sigma_old/sigma;
        alpha = c_alpha /sigma;

        % G2: gradient of L(x,y) with respect to y for x
        [Grad2, Grad1] = inner_prod_matvec(G_tr_class,x,true);
        % G3: gradient of L(x,y) with respect to y for x_old
        Grad3 = (c./trace_kernels).*inner_prod_matvec(G_tr_class,x_old);
        % acceleration step
        grad_y = (1+theta)*((c./trace_kernels).*Grad2)-theta*(Grad3);
        % project onto the simplex
        ytilde = projsplx(y+sigma*grad_y);

        % computing the gradient of L(x,y) with respect to x
        grad_x = -2*ones(m,1)+2*(Grad1)*((c./trace_kernels).*ytilde)+2*reg_param*x;
        xtilde = x - tau * grad_x;
        % project onto the feasible set of x
        xtilde = proj_pos_plane(xtilde,Ytrain);
        xdiff = xtilde-x; ydiff = ytilde-y;
        [Grad2_diff, Grad1_diff] = inner_prod_matvec(G_tr_class,xdiff,true);
        E = sum( (c./trace_kernels).* ytilde .*Grad2_diff);
        [Grad2_new, Grad1_new] = inner_prod_matvec(G_tr_class,xtilde,true);
        aux = (c./trace_kernels).* (Grad2_new - Grad2);
        E = E + xdiff'* reg_param * xdiff - (1-delta)*norm(xdiff)^2/(2*tau)+norm(aux)^2/2/alpha -...
            ((1-delta)/sigma - theta * alpha_old)/2 * norm(ydiff)^2 ;
        if E<=0
            break;
        else
            inner(iter)=inner(iter)+1;
            tau = tau*eta;
        end
    end
    % compute G_xi
    if iter == 50
        % x_curr = x_cum/TK; y_curr = y_cum/TK;
        x_curr = x; y_curr = y;
        % compute the smoothed duality gap
        % x part
        qp_Q = (2 * reg_param + 2 * xi) * eye(m);
        for j = 1:length(y_curr)
            qp_Q = qp_Q + 2 * y_curr(j) * c / trace_kernels(j) * G_tr_class{j,1};
        end
        qp_Q = (qp_Q + qp_Q') / 2;
        qp_q = -2 * ones(m,1) - 2 * xi * x_curr;

        % constraints: x >= 0, Ytrain' * x = 0
        Aeq = Ytrain(:)';
        beq = 0;
        options = optimoptions('quadprog', 'Display', 'off');

        [x_opt, fval, exitflag, output] = quadprog( ...
            qp_Q, qp_q, [], [], Aeq, beq,zeros(m,1), [],[], options);
        G_xpart = fval + xi * norm(x_curr)^2;
        G_xpart = - G_xpart;
        % y part
        [Grad2, Grad1] = inner_prod_matvec(G_tr_class,x_curr,true);
        g_xx= (c./trace_kernels).*Grad2;
        tmp_vec = y_curr + g_xx/2/xi;
        tmp_proj =  projsplx(tmp_vec);
        G_ypart = xi * norm(tmp_proj - tmp_vec)^2 - norm(g_xx)^2/4/xi - g_xx' * y_curr;
        G_ypart = - G_ypart;
        G_ypart = G_ypart + reg_param * norm(x_curr)^2 - 2 * ones(m,1)'*x_curr;
        % sum the x part and y part
        G_xi_old = G_xpart + G_ypart;
        fprintf('Iter: %d, smoothed duality gap: %.4e\n', iter, G_xi_old);
    end

    if mod(iter,K)==0
        fprintf('Iter: %d, line search: %d\n', iter, inner(iter));
    end
    gamma_old = gamma;
    gamma = gamma*(1+mu*tau);
    if non_mono == 0
        tau_new = tau*sqrt(gamma_old/gamma);
    else
        tau_new = min( tau*sqrt(gamma_old/gamma *(1+tau/tau_old)), 10);
    end
    sigma_old = sigma; tau_old = tau;
    tau = tau_new;alpha_old = alpha;

    x_old = x; y = ytilde; x = xtilde;
    x_cum = x_cum + sigma/sigma_0 * x;
    y_cum = y_cum + sigma/sigma_0 * y; TK= TK + sigma/sigma_0;
    if mod(iter,K)==0
        % xi = 1/s_count * 4;
        % x_curr = x_cum/TK; y_curr = y_cum/TK;
        x_curr = x; y_curr = y;
        % compute the smoothed duality gap
        % x part
        qp_Q = (2 * reg_param + 2 * xi) * eye(m);
        for j = 1:length(y_curr)
            qp_Q = qp_Q + 2 * y_curr(j) * c / trace_kernels(j) * G_tr_class{j,1};
        end
        qp_Q = (qp_Q + qp_Q') / 2;
        qp_q = -2 * ones(m,1) - 2 * xi * x_curr;

        % constraints: x >= 0, Ytrain' * x = 0
        Aeq = Ytrain(:)'; beq = 0;
        options = optimoptions('quadprog', 'Display', 'off');
        [x_opt, fval, exitflag, output] = quadprog( ...
            qp_Q, qp_q, [], [], Aeq, beq,zeros(m,1), [],[], options);
        G_xpart = fval + xi * norm(x_curr)^2;
        G_xpart = - G_xpart;
        % y part
        [Grad2, Grad1] = inner_prod_matvec(G_tr_class,x_curr,true);

        g_xx= (c./trace_kernels).*Grad2;
        tmp_vec = y_curr + g_xx/2/xi;
        tmp_proj =  projsplx(tmp_vec);
        G_ypart = xi * norm(tmp_proj - tmp_vec)^2 - norm(g_xx)^2/4/xi - g_xx' * y_curr;
        G_ypart = - G_ypart;
        G_ypart = G_ypart + reg_param * norm(x_curr)^2 - 2 * ones(length(x_curr),1)'*x_curr;
        % sum the x part and y part
        G_xi = G_xpart + G_ypart;
        fprintf('Iter: %d, smoothed duality gap: %.4e\n', iter, G_xi);
        % restart
        if G_xi < G_xi_old * 0.5
            fprintf('perform adaptive restart \n');
            % x_cum = zeros(m,1); y_cum = zeros(n,1); TK=0;
            x = x_curr;y = y_curr; x_old = x;
            G_xi_old = G_xi;
            % eta = 0.7;
            tau = 1/(L_xx+L_yx)*10;
            % gamma = 1;
            sigma_old = gamma*tau;
            % c_alpha = 0.4; delta = 0.5;
            %% be careful about alpha and beta
            alpha_old = c_alpha /sigma_old;
            % because we have both sigma_{k-1} and sigma_k in each iterate
            tau_old = tau;
        end
    end
    epoch_counter = epoch_counter+1;
    time_apd = time_apd+toc;
    rel_err_x(epoch_counter,1) = norm(x-optsol)/norm_optsol;
    time_period_apd(epoch_counter,1) = time_apd;
    iter_epoch(epoch_counter,1) = iter;
    %fprintf(...
    %    'Iteration    Time    Rel. Error Solution   Rel. Error Lagrange\n');
    %fprintf('%d    %9.4f       %9.1e         %9.1e\n',iter,time_period_apd(epoch_counter),...
    %    rel_err_x(epoch_counter),rel_err_lagrange(epoch_counter));
end
tic;
epoch_counter = epoch_counter+1;
rel_err_x(epoch_counter,1) = norm(x-optsol)/norm_optsol;
time_apd = time_apd+toc;
time_period_apd(epoch_counter,1) = time_apd;
iter_epoch(epoch_counter,1) = iter;
fprintf(...
    'Iteration    Time    Rel. Error Solution\n');
fprintf('%d    %9.4f       %9.1e\n',iter,time_period_apd(epoch_counter),...
    rel_err_x(epoch_counter));
end
