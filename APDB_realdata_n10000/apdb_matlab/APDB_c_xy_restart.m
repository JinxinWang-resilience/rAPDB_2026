function [rel_infeas_err,rel_subopt_err,time_period,iter_epoch,oracle]=...
    APDB_c_xy_restart(input,optval,x0,y0,stop_criteria,max_iter,non_mono,K)
A = input{1,1};
Q = input{2,1};
b = input{3,1};
d = input{4,1};
c = input{5,1};
sc = 0;
[n,m] = size(d); % number of constraints
optval_rel = abs(optval)+1;  % used for evaluation
%-------------------------------------%
disp('**********************************************************')
disp('APDB-xy-restarts')
disp('**********************************************************')
epoch=1;
epoch_counter = 0;
%------------ step-size Parmeters------%
eta = 0.7;
tau = 3e-1;
gamma = 3;gamma_0 = 1;
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
y = y0;x = x0;
x_cum = zeros(m,1);y_cum = zeros(n,1); sigma_0 = sigma_old; TK = 0;

iter = 0;
orc = 0; % oracle
elapsed_time = 0;
inner = zeros(max_iter,1); % to record the number of line search in each iteration
G_x = A * x + b;
for j=1:n
    G_x = G_x+(Q{j,1}*x+d(j,:)')*y(j);
end
orc = orc+1;
% for j=1:n
%    G_y(j,1) = 0.5*x'*Q{j,1}*x+d(j,:)*x-c(j);
% end
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
        % E = 1/2/alpha * norm(aux1)^2 + 1/2/beta * norm(aux2)^2 + (delta-1)/2/sigma * norm(ydiff)^2  - ((1-delta)/tau - theta * (alpha_old+beta_old))/2 * norm(xdiff)^2;
        E = 1/2/alpha * norm(aux1)^2 + 1/2/beta * norm(aux2)^2 + (delta-1)/2/sigma * norm(ydiff)^2  - ((1-delta)/tau - theta * (alpha_old+beta_old))/2 * norm(xdiff)^2;
        % E = 1/2/alpha * norm(aux1)^2 + 1/2/beta * norm(aux2)^2 - ((1-delta)/tau - theta * (alpha_old+beta_old))/2 * norm(xdiff)^2;
        if E<=0
            break;
        else
            inner(iter)=inner(iter)+1;
            tau = tau * eta;
        end
    end
    if mod(iter,500)==0
        fprintf('Iter: %d, line search: %d\n', iter, inner(iter));
    end
    gamma_old = gamma;
    gamma = gamma*(1+mu*tau);
    % line search
    if non_mono == 0
        tau_new = tau*sqrt(gamma_old/gamma);
    else
        tau_new = min( tau*sqrt(gamma_old/gamma *(1+tau/tau_old)), 1);
    end
    % tau_new = tau*sqrt(gamma_old/gamma);
    % tau_new = min(tau*sqrt(gamma_old/gamma *(1+tau/tau_old)), 2);
    sigma_old = sigma;
    tau_old = tau;
    alpha_old = alpha;
    beta_old = beta;
    tau = tau_new;
    
    y = ytild;
    x = xtild;
    x_cum = x_cum + sigma/sigma_0 * x; y_cum = y_cum + sigma/sigma_0 * y; TK= TK + sigma/sigma_0;
    % x_cum = x_cum + x; y_cum = y_cum + y; TK= TK + 1;
    %% restarts
    if mod(iter,K) == 0
        fprintf('perform fixed-frequency restart \n');
        % xavg = x_cum/TK; yavg = y_cum/TK;
        xavg = x; yavg = y;
        
        % initialize
        x = xavg;y = yavg;
        x_cum = zeros(m,1);y_cum = zeros(n,1);TK = 0;
        
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
    elapsed_time = elapsed_time + toc;
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