function [rel_infeas_err,rel_subopt_err,time_period,iter_epoch,oracle]=...
    APDB_c_lr_restart(input,optval,x0,y0,stop_criteria,max_iter,non_mono,K)
%************************************************************
% IMPORTANT: optimal solution is used to measure the accuracy of the
% solution found
%
% Written by Erfan Yazdandoost Hamedani, created on 14 November 2018.
%
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
[n,m] = size(d);
optval_rel = abs(optval)+1;
%-------------------------------------%
disp('**********************************************************')
disp('APDB-yx-restarts')
disp('**********************************************************')
epoch=1;
epoch_counter = 0;
%------------ step-size Parmeters------%
eta = 0.7;
tau = 3e-1;
gamma = 1;
c_alpha = 0.4; delta = 0.5;
sigma_old = gamma*tau;
tau_old = tau;
alpha_old = c_alpha /sigma_old;
mu = sc;
%------ Initialization ----------------%
rel_subopt_err = [];
rel_infeas_err = [];
y = y0;
x = x0;
x_cum = zeros(m,1);y_cum = zeros(n,1); sigma_0 = sigma_old; TK = 0;
iter = 0;
orc = 0;
elapsed_time = 0;
inner = zeros(max_iter,1);
for j=1:n
    G_y(j,1) = 0.5*x'*Q{j,1}*x+d(j,:)*x-c(j);
end
%---------- Main Algorithm ------------%
while iter<max_iter
    tic;
    iter = iter+1;
    G_y_old = G_y;
    for j=1:n
        G_y(j,1) = 0.5*x'*Q{j,1}*x+d(j,:)*x-c(j);
    end
    while true
        sigma = gamma*tau;
        theta = sigma_old/sigma;
        alpha = c_alpha /sigma;
        ytild = max(y+sigma*((1+theta)*G_y-theta*G_y_old),0);
        G_x = A*x+b;
        for j=1:n
            G_x = G_x+(Q{j,1}*x+d(j,:)')*ytild(j);
        end
        orc = orc+1;
        xtild = max(min(x-tau*G_x,10),-10);
        xdiff = xtild-x;
        ydiff = ytild-y;
        E = 0;
        aux = 0;
        for j=1:n
            E = E+((xdiff)'*Q{j,1}*(xdiff))*ytild(j);
            aux = aux+norm(0.5*xtild'*Q{j,1}*xtild-0.5*x'*Q{j,1}*x+d(j,:)*(xdiff))^2;
        end
        E = E/2 + xdiff'*A*xdiff/2 - (1-delta)*norm(xdiff)^2/(2*tau)+aux/2/alpha - ((1-delta)/sigma - theta * alpha_old)/2 * norm(ydiff)^2 ;
        if E<=0
            break;
        else
            inner(iter)=inner(iter)+1;
            tau = tau*eta;
        end
    end
    if mod(iter,500)==0
        fprintf('Iter: %d, line search: %d\n', iter, inner(iter));
    end
    gamma_old = gamma;
    gamma = gamma*(1+mu*tau);
    if non_mono == 0
        tau_new = tau*sqrt(gamma_old/gamma);
    else
        tau_new = min( tau*sqrt(gamma_old/gamma *(1+tau/tau_old)), 1);
    end
    sigma_old = sigma;
    tau_old = tau;
    tau = tau_new;
    alpha_old = alpha;
    
    y = ytild;x = xtild;
    x_cum = x_cum + sigma/sigma_0 * x; y_cum = y_cum + sigma/sigma_0 * y; TK= TK + sigma/sigma_0;
    % x_cum = x_cum + x; y_cum = y_cum + y; TK= TK + 1;
    
    %% restarts
    if mod(iter,K) == 0
        fprintf('perform fixed-frequency restart \n');
        % xavg = x_cum/TK; yavg = y_cum/TK;
        xavg = x; yavg = y;
        
        % initialize
        x = xavg; y = yavg;
        x_cum = zeros(m,1);y_cum = zeros(n,1);TK = 0;
        
        % eta = 0.7;
        tau = 3e-1;
        % gamma = 1;
        sigma_old = gamma*tau;
        % c_alpha = 0.4; delta = 0.5;
        %% be careful about alpha and beta
        alpha_old = c_alpha /sigma_old;
        
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
        rel_subopt_err(epoch_counter,1) = abs(subopt-optval)/abs(optval_rel);
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