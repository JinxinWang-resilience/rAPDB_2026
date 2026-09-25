%************************************************************************
% Important: CVX is required http://cvxr.com/cvx/download/
%
% Example of implementing various algorithms
% Mosek, EGM, APDB_yx, rAPDB_yx, rAPDB_yx_ada
% on kernel matrix learning problem.
%*************************************************************************
clear; clc
addpath('C:\Users\jwang195\Downloads\cvx-w64\cvx');
cvx_setup;

seed = 17;
rng(seed,'twister');

%% load data
data = 'sido0_train';
load(data);

nnn = 1000;
X = X(1:nnn, :);

mu_tmp = mean(X, 1);
sigma_tmp = std(X, 0, 1);
% sigma_tmp(sigma_tmp == 0) = 1;   % avoid division by zero
X = (X - mu_tmp) ./ sigma_tmp;

nan_cols = any(isnan(X), 1);
X = X(:, ~nan_cols);

Y = load('sido0_train.targets');
Y = Y(1:nnn,:);

reg_param = 1;

display(['Data ', data, ' has been loaded!']);
display(['size of data = ', num2str(size(X,1)), '*', num2str(size(X,2))]);

%% Create kernel matrices
[Ytrain, Ytest, Kernel_tr_class, Kernel_test_class, ...
    G_tr_class, G_tr_max_norm, trace_kernels] = kernel_data_generator(X,Y);

num_kern = length(trace_kernels);
input = {G_tr_class; Kernel_tr_class; Kernel_test_class; ...
         G_tr_max_norm; Ytrain; Ytest; trace_kernels};

%% CVX
c = sum(trace_kernels);
train_size = length(Ytrain);

display('Obtaining the optimal solution using CVX...')
tic;
cvx_begin quiet
    cvx_solver mosek
    cvx_precision high
    variables alphastar(train_size,1) t(1,1);
    dual variable mustar
    minimize(-2*alphastar'*ones(train_size,1) + reg_param*sum_square(alphastar) + c*t);
    subject to
        mustar: t*ones(num_kern,1) >= inner_prod_matvec(G_tr_class,alphastar)./trace_kernels;
        alphastar'*Ytrain == 0;
        alphastar >= 0;
cvx_end
time_cvx = toc;

optval = cvx_optval;
optsol = alphastar;

fprintf('CVX time = %7.4f, optimal value = %9.1e\n', time_cvx, optval);

%% Initial point and parameters
x0 = zeros(length(Ytrain),1);
y0 = zeros(num_kern,1);

max_iter = 8e3;
stop_criteria = 1e-8;

%% -------- Run EGM once --------
disp('Running EGM ...')
[rel_err_egm, time_egm, iter_egm] = ...
    EGM(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter);

%% -------- Run three algorithms with non_monotone = 0 --------
non_monotone = 0;
K1 = 600;
K  = 600;

disp('Running APDB_yx, rAPDB_yx, rAPDB_yx_ada with non_monotone = 0 ...')

[rel_err_apdb_0, time_apdb_0, iter_apdb_0] = ...
    APDB_yx(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter, non_monotone);

[rel_err_rapdb_0, time_rapdb_0, iter_rapdb_0] = ...
    rAPDB_yx(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter, K1, non_monotone);

[rel_err_rapdb_ada_0, time_rapdb_ada_0, iter_rapdb_ada_0] = ...
    rAPDB_yx_ada(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter, K, non_monotone);

%% -------- Run three algorithms with non_monotone = 1 --------
non_monotone = 1;
K1 = 100;
K  = 100;

disp('Running APDB_yx, rAPDB_yx, rAPDB_yx_ada with non_monotone = 1 ...')

[rel_err_apdb_1, time_apdb_1, iter_apdb_1] = ...
    APDB_yx(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter, non_monotone);

[rel_err_rapdb_1, time_rapdb_1, iter_rapdb_1] = ...
    rAPDB_yx(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter, K1, non_monotone);

[rel_err_rapdb_ada_1, time_rapdb_ada_1, iter_rapdb_ada_1] = ...
    rAPDB_yx_ada(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter, K, non_monotone);

%% -------- Plot all curves in one figure --------
figure;

% colors
c1 = [0.000, 0.447, 0.741];
c2 = [0.850, 0.325, 0.098];
c3 = [0.929, 0.694, 0.125];
c4 = [0.494, 0.184, 0.556];
c5 = [0.466, 0.674, 0.188];
c6 = [0.301, 0.745, 0.933];
c7 = [0.635, 0.078, 0.184];

%% subplot 1: iteration
subplot(2,1,1);
semilogy(iter_egm,           rel_err_egm,           '-',  'Color', c1, 'LineWidth', 1.6); hold on;
semilogy(iter_apdb_0,        rel_err_apdb_0,        '-',  'Color', c2, 'LineWidth', 1.6);
semilogy(iter_rapdb_0,       rel_err_rapdb_0,       '-',  'Color', c3, 'LineWidth', 1.6);
semilogy(iter_rapdb_ada_0,   rel_err_rapdb_ada_0,   '-',  'Color', c4, 'LineWidth', 1.6);

semilogy(iter_apdb_1,        rel_err_apdb_1,        '--', 'Color', c2, 'LineWidth', 1.6);
semilogy(iter_rapdb_1,       rel_err_rapdb_1,       '--', 'Color', c3, 'LineWidth', 1.6);
semilogy(iter_rapdb_ada_1,   rel_err_rapdb_ada_1,   '--', 'Color', c4, 'LineWidth', 1.6);

grid on;
xlabel('Iteration: k');
ylabel('$\|x_k-x^*\|/\|x^*\|$','Interpreter','latex');
legend({'EGM', ...
        'APDB-yx (mono)', 'rAPDB-yx (mono)', 'rAPDB-yx-ada (mono)', ...
        'APDB-yx (nonmono)', 'rAPDB-yx (nonmono)', 'rAPDB-yx-ada (nonmono)'}, ...
        'FontSize', 10, 'Location', 'best');
title('Relative error vs iteration');

%% subplot 2: time
subplot(2,1,2);
semilogy(time_egm,           rel_err_egm,           '-',  'Color', c1, 'LineWidth', 1.6); hold on;
semilogy(time_apdb_0,        rel_err_apdb_0,        '-',  'Color', c2, 'LineWidth', 1.6);
semilogy(time_rapdb_0,       rel_err_rapdb_0,       '-',  'Color', c3, 'LineWidth', 1.6);
semilogy(time_rapdb_ada_0,   rel_err_rapdb_ada_0,   '-',  'Color', c4, 'LineWidth', 1.6);

semilogy(time_apdb_1,        rel_err_apdb_1,        '--', 'Color', c2, 'LineWidth', 1.6);
semilogy(time_rapdb_1,       rel_err_rapdb_1,       '--', 'Color', c3, 'LineWidth', 1.6);
semilogy(time_rapdb_ada_1,   rel_err_rapdb_ada_1,   '--', 'Color', c4, 'LineWidth', 1.6);

grid on;
xlabel('Time (s)');
ylabel('$\|x_k-x^*\|/\|x^*\|$','Interpreter','latex');
legend({'EGM', ...
        'APDB-yx (mono)', 'rAPDB-yx (mono)', 'rAPDB-yx-ada (mono)', ...
        'APDB-yx (nonmono)', 'rAPDB-yx (nonmono)', 'rAPDB-yx-ada (nonmono)'}, ...
        'FontSize', 10, 'Location', 'best');
title('Relative error vs time');