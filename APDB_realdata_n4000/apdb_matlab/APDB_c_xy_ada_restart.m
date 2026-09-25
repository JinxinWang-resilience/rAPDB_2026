function [rel_infeas_err,rel_subopt_err,time_period,iter_epoch,oracle]=...
    APDB_c_xy_ada_restart(input,optval,x0,y0,stop_criteria,max_iter,non_mono)
%************************************************************
% IMPORTANT: optimal solution is used to measure the accuracy of the
% solution found

% Adaptive restart
% Revised the line search part compared with APDB_c.m

% The algorithm is specified to solve Quadratic Constrained
% Quadratic Programming (QCQP)
%
% The step-sizes are constant according to
% Theorem 2.1 part (I) in the paper https://arxiv.org/pdf/1803.01401.pdf
%************************************************************
% min_{x} 0.5*x'*A*x+b'*x
% s.t.    0.5*x'*Q_i*x+d_i'*x-c_i<=0, for i=1:m,
%         -10<=x<=10
%************************************************************
%---------- Unfolding Input ----------%
A = input{1,1};
Q = input{2,1};
b = input{3,1};
d = input{4,1};
c = input{5,1};
sc = 0;
[n,m] = size(d); % n: the number of constraints
optval_rel = abs(optval)+1;
%-------------------------------------%
disp('**********************************************************')
disp('APDB-xy-adaptive restart')
disp('**********************************************************')
epoch=1;
epoch_counter = 0;
%------------ step-size Parmeters------%
eta = 0.7;
tau = 3e-1;
gamma = 3; gamma_0 = 1;
sigma_old = gamma*tau;
c_alpha = 0.25; c_beta = 0.3; delta = 0.4;
%% be careful about alpha and beta
alpha_old = c_alpha /tau;
beta_old = gamma_0 * c_beta/sigma_old; % because gamma_0 = 1

% because we have both sigma_{k-1} and sigma_k in each iterate
tau_old = tau;
mu = sc;
%------ Initialization ----------------%
rel_subopt_err = [];
rel_infeas_err = [];
y = y0;
x = x0;
iter = 0;
orc = 0;
elapsed_time = 0;
inner = zeros(max_iter,1); % to record the number of line search in each iteration
G_x = A * x + b;
for j=1:n
    G_x = G_x+(Q{j,1}*x+d(j,:)')*y(j);
end
orc = orc+1;
xi = 0.04;
x_cum = zeros(m,1);y_cum = zeros(n,1); sigma_0 = sigma_old; TK = 0;

%---------- Main Algorithm ------------%
while iter<max_iter
    tic;
    iter = iter+1;
    G_x_old = G_x;
    while true   % line search
        sigma = gamma*tau;
        theta = sigma_old/sigma;
        alpha = c_alpha /tau;
        beta = gamma_0 * c_beta/sigma; % because gamma_0 = 1
        %% to update first x, then y
        G_x = A*x+b;
        for j=1:n
            G_x = G_x+(x' * Q{j,1} +d(j,:))'*y(j);
        end

        xtild = max(min(x-tau*((1+theta)*G_x - theta*G_x_old),10),-10);
        xdiff = xtild-x;
        for j=1:n
            G_y(j,1) = 0.5*xtild'*Q{j,1}*xtild+d(j,:)*xtild-c(j);
        end
        orc = orc+1;
        ytild = max(y+sigma*(G_y),0);
        ydiff = ytild-y;
        % the new function E_k
        aux1 = zeros(m,1);
        aux2 = A * xdiff;
        for j=1:n
            aux1 = aux1 + (xtild' * Q{j,1} + d(j,:))' *ydiff(j);
            aux2 = aux2 + (Q{j,1}*xdiff) * y(j);
        end
        E = 1/2/alpha * norm(aux1)^2 + 1/2/beta * norm(aux2)^2 + (delta-1)/2/sigma * norm(ydiff)^2  ...
            - ((1-delta)/tau - theta * (alpha_old+beta_old))/2 * norm(xdiff)^2;
        % E = 1/2/alpha * norm(aux1)^2 + 1/2/beta * norm(aux2)^2  - ((1-delta)/tau -...
        % theta * (alpha_old+beta_old))/2 * norm(xdiff)^2;
        if E<=0
            break;
        else
            inner(iter)=inner(iter)+1;
            tau = tau * eta;
        end
    end
    %% adaptive restart
    if iter==100
        % xi = 1/s_count * 4;
        x_curr = x_cum/TK;y_curr = y_cum/TK;
        x_curr = x; y_curr = y;
        % compute the smoothed duality gap
        % x part
        qp_Q = A + 2 * xi * eye(m);
        for j=1:n
            qp_Q = qp_Q + y_curr(j) * Q{j,1};
        end
        qp_Q = (qp_Q + qp_Q')/2;
        qp_q = b - 2 * xi * x_curr;
        for j=1:n
            qp_q = qp_q + y_curr(j) * d(j,:)';
        end
        % solve QP
        lb = -10 * ones(m,1);       % lower bound
        ub =  10 * ones(m,1);       % upper bound
        options = optimoptions('quadprog','Display','off');
        [x_opt, fval, exitflag, output] = quadprog(qp_Q, qp_q, [], [], [], [], lb, ub, [], options);
        G_xpart = fval + xi * norm(x_curr)^2;
        for j=1:n
            G_xpart = G_xpart - y_curr(j)*c(j);
        end
        G_xpart = - G_xpart;
        % y part
        g_x = zeros(n,1);
        for j = 1:n
            g_x(j) = 0.5*x_curr'*Q{j,:}*x_curr+d(j,:)*x_curr-c(j);
        end
        tmp_vec = y_curr + g_x/2/xi;
        G_ypart = xi * sum((min(tmp_vec,0)).^2) - norm(g_x)^2/4/xi - g_x' * y_curr;
        G_ypart = - G_ypart;
        G_ypart = G_ypart + 0.5*x_curr'*A'*x_curr+b'*x_curr;
        % sum the x part and y part
        G_xi_old = G_xpart + G_ypart;
        fprintf('Iter: %d, smoothed duality gap: %.4e\n', iter, G_xi_old);
    end

    if mod(iter,500)==0
        fprintf('Iter: %d, line search: %d\n', iter, inner(iter));
    end
    gamma_old = gamma;
    gamma = gamma*(1+mu*tau);
    % line search
    % tau_new = tau*sqrt(gamma_old/gamma);
    % non-monotonic line search
    if non_mono == 0
        tau_new = tau*sqrt(gamma_old/gamma);
    else
        tau_new = min( tau*sqrt(gamma_old/gamma *(1+tau/tau_old)), 1);
    end
    sigma_old = sigma;
    tau_old = tau;
    alpha_old = alpha;
    beta_old = beta;
    tau = tau_new;

    y = ytild;x = xtild;

    x_cum = x_cum + sigma/sigma_0 * x; y_cum = y_cum + sigma/sigma_0 * y; TK= TK + sigma/sigma_0;
    % x_cum = x_cum + x; y_cum = y_cum + y; TK= TK + 1;
    elapsed_time = elapsed_time + toc;

    if mod(iter,500)==0
        % xi = 1/s_count * 4;
        % x_curr = x_cum/TK;y_curr = y_cum/TK;
        x_curr = x; y_curr = y;
        % compute the smoothed duality gap
        % x part
        qp_Q = A + 2 * xi * eye(m);
        for j=1:n
            qp_Q = qp_Q + y_curr(j) * Q{j,1};
        end
        qp_Q = (qp_Q + qp_Q')/2;
        qp_q = b - 2 * xi * x_curr;
        for j=1:n
            qp_q = qp_q + y_curr(j) * d(j,:)';
        end
        % solve QP
        lb = -10 * ones(m,1);       % lower bound
        ub =  10 * ones(m,1);       % upper bound
        options = optimoptions('quadprog','Display','off');
        [x_opt, fval, exitflag, output] = quadprog(qp_Q, qp_q, [], [], [], [], lb, ub, [], options);
        G_xpart = fval + xi * norm(x_curr)^2;
        for j=1:n
            G_xpart = G_xpart - y_curr(j)*c(j);
        end
        G_xpart = - G_xpart;
        % y part
        g_x = zeros(n,1);
        for j = 1:n
            g_x(j) = 0.5*x_curr'*Q{j,:}*x_curr+d(j,:)*x_curr-c(j);
        end
        tmp_vec = y_curr + g_x/2/xi;
        G_ypart = xi * sum((min(tmp_vec,0)).^2) - norm(g_x)^2/4/xi - g_x' * y_curr;
        G_ypart = - G_ypart;
        G_ypart = G_ypart + 0.5*x_curr'*A'*x_curr+b'*x_curr;
        % sum the x part and y part
        G_xi = G_xpart + G_ypart;
        fprintf('Iter: %d, smoothed duality gap: %.4e\n', iter, G_xi);
        % restart
        if G_xi < G_xi_old * 0.5
            fprintf('perform adaptive restart \n');
            x_cum = zeros(m,1);y_cum = zeros(n,1);TK=0;
            x = x_curr; y = y_curr;
            G_xi_old = G_xi;
            % eta = 0.7;
            tau = 3e-1;
            % gamma = 1;
            sigma_old = gamma*tau;
            % c_alpha = 0.25; c_beta = 0.3; delta = 0.4;
            %% be careful about alpha and beta
            alpha_old = c_alpha /tau;
            beta_old = gamma_0 * c_beta/sigma_old; % because gamma_0 = 1

            % because we have both sigma_{k-1} and sigma_k in each iterate
            tau_old = tau;
            G_x = A * x + b;
            for j=1:n
                G_x = G_x+(Q{j,1}*x+d(j,:)')*y(j);
            end
        end
    end

    if mod(iter,epoch) == 0
        epoch_counter = epoch_counter+1;
        time_period(epoch_counter,1) = elapsed_time;
        oracle(epoch_counter,1) = orc;
        subopt = 0.5*x'*A'*x+b'*x;
        infeas = 0;
        for l=1:n
            infeas = infeas+pos(0.5*x'*Q{l,:}*x+d(l,:)*x-c(l));
        end
        rel_subopt_err(epoch_counter,1) = abs(subopt-optval)/optval_rel;
        rel_infeas_err(epoch_counter,1) = infeas/n;
        iter_epoch(epoch_counter,1) = iter;
        if max(abs(subopt-optval)/abs(optval_rel), infeas/n)<stop_criteria
            fprintf(...
                'Iteration    Time    Rel. Infeas error   Rel. Subopt error\n');
            fprintf('%d    %9.4f       %9.1e         %9.1e\n',iter,time_period(epoch_counter),...
                rel_infeas_err(epoch_counter),rel_subopt_err(epoch_counter));
            return;
        end
    end
end
fprintf(...
    'Iteration    Time    Rel. Infeas error   Rel. Subopt error\n');
fprintf('%d    %9.4f       %9.1e         %9.1e\n',iter,time_period(epoch_counter),...
    rel_infeas_err(epoch_counter),rel_subopt_err(epoch_counter));
end