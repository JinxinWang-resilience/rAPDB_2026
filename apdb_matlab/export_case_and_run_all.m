% Export one shared QCQP case and run all MATLAB solvers on it.
% Outputs:
%   outputs/python_shared_case.mat
%   outputs/matlab_shared_results.mat

clear;
clc;

outdir = 'outputs';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

seed = 123;
rng(seed, 'twister');

numsim = 1; %#ok<NASGU>
n = 10;
m = 90;
x0 = rand(m, 1);
y0 = zeros(n, 1);

fprintf('Generating one shared QCQP case...\n');

S = orth(randn(m, m));
D = rand(m - 1, 1) * 100;
A = S' * diag([D; 1e-10]) * S;
A = (A + A') / 2;
b = randn(m, 1);

sc = min(eig(A));
if sc < 1e-8
    sc = 0;
end

d = randn(n, m);
Q = cell(n, 1);
for j = 1:n
    S = orth(randn(m, m));
    D = rand(m - 1, 1) * 100;
    Q_tmp = S' * diag([D; 1e-10]) * S;
    Q{j, 1} = (Q_tmp + Q_tmp') / 2;
end
c = rand(n, 1);

fprintf('Running CVX/MOSEK reference solve...\n');
tic;
cvx_begin quiet
cvx_precision high
variable xstar(m,1)
dual variable ystar{n}
constraint = cvx(zeros(n,1));
for l = 1:n
    constraint(l) = 0.5 * xstar' * Q{l,1} * xstar + d(l,:) * xstar - c(l);
end
minimize(0.5 * xstar' * A * xstar + b' * xstar);
subject to
for l = 1:n
    constraint(l) <= 0 : ystar{l}; %#ok<NASGU>
end
xstar <= 10 * ones(m,1);
xstar >= -10 * ones(m,1);
cvx_end
mosek_time = toc; %#ok<NASGU>
Total_time_Mosek = mosek_time; %#ok<NASGU>
optval = cvx_optval;

save(fullfile(outdir, 'python_shared_case.mat'), 'A', 'Q', 'b', 'd', 'c', 'sc', 'x0', 'y0', 'optval');
fprintf('Saved shared case: %s\n', fullfile(outdir, 'python_shared_case.mat'));

input = {A; Q; b; d; c; sc};
max_iter = 5e4;
epsilon = 1e-8;

fprintf('Running EGM (with tau tuning)...\n');
opts_egm.epoch = 1;
opts_egm.verbose = 300;
opts_egm.tau_grid = [1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3];
opts_egm.tune_iter = 2000;
[best_tau_egm{1,1}, best_out_egm, tune_table_egm{1,1}] = ... %#ok<NASGU>
    EGM_tune_tau(input, optval, x0, y0, epsilon, max_iter, opts_egm);

rel_infeas_err_egm{1,1} = best_out_egm.rel_infeas_err;
rel_subopt_egm{1,1} = best_out_egm.rel_subopt_err;
time_period_egm{1,1} = best_out_egm.time_period;
iter_epoch_egm{1,1} = best_out_egm.iter_epoch;
oracle_egm{1,1} = best_out_egm.oracle;

nonmono_list = [0, 1];
for mode_idx = 1:length(nonmono_list)
    whether_nonmono = nonmono_list(mode_idx);
    if whether_nonmono == 0
        K = 2000;
        K1 = 800;
    else
        K = 1000;
        K1 = 500;
    end

    fprintf('Running mode whether_nonmono = %d...\n', whether_nonmono);

    [rel_infeas_1{mode_idx,1}, rel_subopt_1{mode_idx,1}, time_period_1{mode_idx,1}, ...
        iter_epoch_1{mode_idx,1}, oracle_1{mode_idx,1}] = ...
        APDB_c_lr(input, optval, x0, y0, epsilon, max_iter, whether_nonmono);

    [rel_infeas_1_ada{mode_idx,1}, rel_subopt_1_ada{mode_idx,1}, time_period_1_ada{mode_idx,1}, ...
        iter_epoch_1_ada{mode_idx,1}, oracle_1_ada{mode_idx,1}] = ...
        APDB_c_lr_ada_restart(input, optval, x0, y0, epsilon, max_iter, whether_nonmono);

    [rel_infeas_r1{mode_idx,1}, rel_subopt_r1{mode_idx,1}, time_period_r1{mode_idx,1}, ...
        iter_epoch_r1{mode_idx,1}, oracle_r1{mode_idx,1}] = ...
        APDB_c_lr_restart(input, optval, x0, y0, epsilon, max_iter, whether_nonmono, K1);

    [rel_infeas_xy_1{mode_idx,1}, rel_subopt_xy_1{mode_idx,1}, time_period_xy_1{mode_idx,1}, ...
        iter_epoch_xy_1{mode_idx,1}, oracle_xy_1{mode_idx,1}] = ...
        APDB_c_xy(input, optval, x0, y0, epsilon, max_iter, whether_nonmono);

    [rel_infeas_xy_r1{mode_idx,1}, rel_subopt_xy_r1{mode_idx,1}, time_period_xy_r1{mode_idx,1}, ...
        iter_epoch_xy_r1{mode_idx,1}, oracle_xy_r1{mode_idx,1}] = ...
        APDB_c_xy_restart(input, optval, x0, y0, epsilon, max_iter, whether_nonmono, K);

    [rel_infeas_xy_ada1{mode_idx,1}, rel_subopt_xy_ada1{mode_idx,1}, time_period_xy_ada1{mode_idx,1}, ...
        iter_epoch_xy_ada1{mode_idx,1}, oracle_xy_ada1{mode_idx,1}] = ...
        APDB_c_xy_ada_restart(input, optval, x0, y0, epsilon, max_iter, whether_nonmono);
end

save(fullfile(outdir, 'matlab_shared_results.mat'), ...
    'mosek_time', 'Total_time_Mosek', ...
    'rel_infeas_err_egm', 'rel_subopt_egm', 'time_period_egm', 'iter_epoch_egm', 'oracle_egm', ...
    'rel_infeas_1', 'rel_subopt_1', 'time_period_1', 'iter_epoch_1', 'oracle_1', ...
    'rel_infeas_1_ada', 'rel_subopt_1_ada', 'time_period_1_ada', 'iter_epoch_1_ada', 'oracle_1_ada', ...
    'rel_infeas_r1', 'rel_subopt_r1', 'time_period_r1', 'iter_epoch_r1', 'oracle_r1', ...
    'rel_infeas_xy_1', 'rel_subopt_xy_1', 'time_period_xy_1', 'iter_epoch_xy_1', 'oracle_xy_1', ...
    'rel_infeas_xy_r1', 'rel_subopt_xy_r1', 'time_period_xy_r1', 'iter_epoch_xy_r1', 'oracle_xy_r1', ...
    'rel_infeas_xy_ada1', 'rel_subopt_xy_ada1', 'time_period_xy_ada1', 'iter_epoch_xy_ada1', 'oracle_xy_ada1');

fprintf('Saved MATLAB all-algorithm results: %s\n', fullfile(outdir, 'matlab_shared_results.mat'));
fprintf('Done.\n');
