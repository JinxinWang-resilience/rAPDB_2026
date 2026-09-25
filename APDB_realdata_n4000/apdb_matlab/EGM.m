function [rel_infeas_err,rel_subopt_err,time_period,iter_epoch,oracle] = ...
    EGM(input,optval,x0,y0,stop_criteria,max_iter,opts)
%************************************************************
% Extragradient Method for QCQP minimax formulation
%
% Solve the primal-dual saddle-point problem
%   min_{x in [lb,ub]} max_{y >= 0}
%       0.5*x'*A*x + b'*x + sum_{i=1}^m y_i*(0.5*x'*Q_i*x + d_i'*x - c_i)
%
% Equivalently, the original QCQP is
%   min_x  0.5*x'*A*x + b'*x
%   s.t.   0.5*x'*Q_i*x + d_i'*x - c_i <= 0,   i=1,...,m.
%
% INPUT
%   input = {A,Q,b,d,c}
%       A : n-by-n matrix
%       Q : cell array, Q{i} is n-by-n
%       b : n-by-1 vector
%       d : m-by-n matrix (or n-by-m, auto-transposed)
%       c : m-by-1 vector
%   optval         : optimal primal objective value, used for reporting only
%   stop_criteria  : stopping tolerance on max(rel_subopt_err, rel_infeas_err)
%   max_iter       : maximum number of iterations
%   opts           : optional struct with fields
%       .tau       : primal stepsize          (default 1e-4)
%       .y_max     : upper bound for y        (default inf, i.e., no cap)
%       .epoch     : record every epoch iters (default 1)
%       .verbose   : print every verbose iters(default 100)
%
% OUTPUT
%   rel_infeas_err : average constraint violation history
%   rel_subopt_err : relative objective error history
%   time_period    : cumulative runtime history
%   iter_epoch     : iteration indices for logged history
%   oracle         : number of operator evaluations used in history
%   x, y           : final primal/dual iterates
%************************************************************

%---------- Unfolding Input ----------%
A = input{1,1};
Q = input{2,1};
b = input{3,1};
d = input{4,1};
c = input{5,1};

x = x0(:);y = y0(:);
b = b(:);c = c(:);

n = length(x);m = length(c);

if size(d,1) ~= m || size(d,2) ~= n
    if size(d,1) == n && size(d,2) == m
        d = d';
    else
        error('Size mismatch: d should be m-by-n (or n-by-m).');
    end
end

optval_rel = abs(optval)+1;

tau = opts.tau;
epoch = opts.epoch;
box_lb = -10;
box_ub = 10;

% project initial point
x = min(max(x, box_lb), box_ub); y = max(y, 0);

%---------- Display ----------%
disp('**********************************************************')
disp('Extragradient for QCQP minimax')
disp('**********************************************************')

%---------- History containers ----------%
rel_subopt_err = [];rel_infeas_err = [];
time_period = [];iter_epoch = [];oracle = [];

iter = 0;orc = 0;epoch_counter = 0;elapsed_time = 0;

%---------- Main loop ----------%
while iter < max_iter
    tic;
    iter = iter + 1;
    
    % Step 1: evaluate operator at current point
    [grad_x, g_val] = saddle_operator(x,y,A,Q,b,d,c);
    orc = orc + 1;
    
    % Predictor
    x_bar = min(max(x - tau*grad_x, box_lb), box_ub);
    y_bar = max(y + tau*g_val, 0);
    
    % Step 2: evaluate operator at predictor point
    [grad_x_bar, g_val_bar] = saddle_operator(x_bar,y_bar,A,Q,b,d,c);
    orc = orc + 1;
    
    % Corrector
    x = min(max(x - tau*grad_x_bar, box_lb), box_ub);
    y = max(y + tau*g_val_bar, 0);
   
    elapsed_time = elapsed_time + toc;
    
    if mod(iter, epoch) == 0
        epoch_counter = epoch_counter + 1;
        time_period(epoch_counter,1) = elapsed_time;
        oracle(epoch_counter,1) = orc;
        iter_epoch(epoch_counter,1) = iter;
        
        subopt = 0.5*x'*(A*x) + b'*x;
        infeas = mean(max(constraint_values(x,Q,d,c), 0));
        
        rel_subopt_err(epoch_counter,1) = abs(subopt - optval) / optval_rel;
        rel_infeas_err(epoch_counter,1) = infeas;
        
        if max(rel_subopt_err(epoch_counter,1), rel_infeas_err(epoch_counter,1)) < stop_criteria
            fprintf('Iteration    Time    Rel. Infeas error   Rel. Subopt error\n');
            fprintf('%d    %9.4f       %9.1e         %9.1e\n', ...
                iter, time_period(epoch_counter), ...
                rel_infeas_err(epoch_counter), rel_subopt_err(epoch_counter));
            return;
        end
    end
end
end

%======================= Local functions =======================%
function [grad_x, g_val] = saddle_operator(x,y,A,Q,b,d,c)
% grad_x = nabla_x L(x,y)
% g_val  = constraint values, i.e., nabla_y L(x,y)

m = length(c);
grad_x = A*x + b;
g_val = zeros(m,1);
for i = 1:m
    qi_x = Q{i}*x;
    g_val(i) = 0.5*x'*qi_x + d(i,:)*x - c(i);
    grad_x = grad_x + y(i)*(qi_x + d(i,:)');
end
end

function g = constraint_values(x,Q,d,c)
m = length(c);
g = zeros(m,1);
for i = 1:m
    g(i) = 0.5*x'*(Q{i}*x) + d(i,:)*x - c(i);
end
end
