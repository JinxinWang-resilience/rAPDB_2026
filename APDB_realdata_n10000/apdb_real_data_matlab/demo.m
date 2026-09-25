%************************************************************************
% Important: CVX is required http://cvxr.com/cvx/download/
%
% Example of implementing various algorithms
% Mosek, EGM, APDB_yx, rAPDB_yx, rAPDB_yx_ada
% on kernel matrix learning problem.
%*************************************************************************
clear; clc
% addpath('C:\Users\jwang195\Downloads\cvx-w64\cvx');
% cvx_setup;

seed = 17;
rng(seed,'twister');

%% output folder
outdir = 'results_kml_single_run';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

%% load data
data = 'sido0_train';
load(data);

nnn = 5000;
X = X(1:nnn, :);

mu_tmp = mean(X, 1);
sigma_tmp = std(X, 0, 1);
sigma_tmp(sigma_tmp < 1e-8) = 1;   % avoid division by zero
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

%% CVX (MOSEK)
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

fprintf('CVX/MOSEK time = %7.4f, optimal value = %9.4e\n', time_cvx, optval);

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
K  = 200;

disp('Running APDB_yx, rAPDB_yx, rAPDB_yx_ada with non_monotone = 1 ...')

[rel_err_apdb_1, time_apdb_1, iter_apdb_1] = ...
    APDB_yx(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter, non_monotone);

[rel_err_rapdb_1, time_rapdb_1, iter_rapdb_1] = ...
    rAPDB_yx(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter, K1, non_monotone);

[rel_err_rapdb_ada_1, time_rapdb_ada_1, iter_rapdb_ada_1] = ...
    rAPDB_yx_ada(input, optsol, optval, reg_param, x0, y0, stop_criteria, max_iter, K, non_monotone);

%% -------- Summary table --------
alg_names = { ...
    'MOSEK(CVX)'; ...
    'EGM'; ...
    'APDB-yx (mono)'; ...
    'rAPDB-yx (mono)'; ...
    'rAPDB-yx-ada (mono)'; ...
    'APDB-yx (nonmono)'; ...
    'rAPDB-yx (nonmono)'; ...
    'rAPDB-yx-ada (nonmono)'};

num_iter = [ ...
    NaN; ...
    iter_egm(end); ...
    iter_apdb_0(end); ...
    iter_rapdb_0(end); ...
    iter_rapdb_ada_0(end); ...
    iter_apdb_1(end); ...
    iter_rapdb_1(end); ...
    iter_rapdb_ada_1(end)];

final_residual = [ ...
    NaN; ...
    rel_err_egm(end); ...
    rel_err_apdb_0(end); ...
    rel_err_rapdb_0(end); ...
    rel_err_rapdb_ada_0(end); ...
    rel_err_apdb_1(end); ...
    rel_err_rapdb_1(end); ...
    rel_err_rapdb_ada_1(end)];

total_time = [ ...
    time_cvx; ...
    time_egm(end); ...
    time_apdb_0(end); ...
    time_rapdb_0(end); ...
    time_rapdb_ada_0(end); ...
    time_apdb_1(end); ...
    time_rapdb_1(end); ...
    time_rapdb_ada_1(end)];

ResultTable = table(alg_names, num_iter, final_residual, total_time, ...
    'VariableNames', {'Algorithm','Iterations','FinalResidual','TimeSeconds'});

disp(' ')
disp('================ Summary Table ================')
disp(ResultTable)

fprintf('\n%-24s %-12s %-16s %-12s\n', 'Algorithm', 'Iterations', 'FinalResidual', 'Time(s)');
fprintf('%s\n', repmat('-',1,72));
for i = 1:height(ResultTable)
    if isnan(ResultTable.Iterations(i))
        iter_str = '-';
    else
        iter_str = sprintf('%.0f', ResultTable.Iterations(i));
    end

    if isnan(ResultTable.FinalResidual(i))
        res_str = '-';
    else
        res_str = sprintf('%.2e', ResultTable.FinalResidual(i));
    end

    fprintf('%-24s %-12s %-16s %-12.2f\n', ...
        ResultTable.Algorithm{i}, ...
        iter_str, ...
        res_str, ...
        ResultTable.TimeSeconds(i));
end

%% -------- Plot all curves in one figure --------
fig = figure('Position',[100,100,1500,550]);

% colors
c1 = [0.000, 0.447, 0.741];
c2 = [0.850, 0.325, 0.098];
c3 = [0.929, 0.694, 0.125];
c4 = [0.494, 0.184, 0.556];

% marker gaps for nonmonotone curves
gap_apdb_iter      = max(1, floor(length(iter_apdb_1)/10));
gap_rapdbada_iter  = max(1, floor(length(iter_rapdb_ada_1)/12));
gap_rapdb_iter     = max(1, floor(length(iter_rapdb_1)/15));

gap_apdb_time      = max(1, floor(length(time_apdb_1)/10));
gap_rapdbada_time  = max(1, floor(length(time_rapdb_ada_1)/12));
gap_rapdb_time     = max(1, floor(length(time_rapdb_1)/15));

%% subplot 1: iteration
subplot(1,2,1);
ax = gca;
ax.FontSize = 16;
ax.LineWidth = 1.2;

p1 = semilogy(iter_egm,         rel_err_egm,         '--', 'Color', c4, 'LineWidth', 3); hold on;
p2 = semilogy(iter_apdb_0,      rel_err_apdb_0,      '-',  'Color', c1, 'LineWidth', 3);
p3 = semilogy(iter_rapdb_ada_0, rel_err_rapdb_ada_0, ':',  'Color', [0.466,0.674,0.188], 'LineWidth', 3);
p4 = semilogy(iter_rapdb_0,     rel_err_rapdb_0,     ':',  'Color', [0.635,0.078,0.184], 'LineWidth', 3);

p5 = semilogy(iter_apdb_1,      rel_err_apdb_1,      '--', 'Color', c1, ...
              'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 9, ...
              'MarkerIndices', 1:gap_apdb_iter:length(iter_apdb_1));
p6 = semilogy(iter_rapdb_ada_1, rel_err_rapdb_ada_1, '-',  'Color', [0.466,0.674,0.188], ...
              'LineWidth', 3, 'Marker', '^', 'MarkerSize', 9, ...
              'MarkerIndices', 1:gap_rapdbada_iter:length(iter_rapdb_ada_1));
p7 = semilogy(iter_rapdb_1,     rel_err_rapdb_1,     '-',  'Color', [0.635,0.078,0.184], ...
              'LineWidth', 3, 'Marker', '+', 'MarkerSize', 9, ...
              'MarkerIndices', 1:gap_rapdb_iter:length(iter_rapdb_1));

grid on
hx = xlabel('Iteration','Interpreter','latex');
hy = ylabel('$\|x^k-x^\star\|/(1+\|x^\star\|)$','Interpreter','latex');
hx.FontSize = 16;
hy.FontSize = 16;

lgd = legend([p1 p2 p3 p4 p5 p6 p7], ...
    {'EGM','APDB-yx (mono)','rAPDB-yx-ada (mono)','rAPDB-yx (mono)', ...
     'APDB-yx (nonmono)','rAPDB-yx-ada (nonmono)','rAPDB-yx (nonmono)'});
lgd.FontSize = 11;
lgd.Location = 'northeast';

%% subplot 2: time
subplot(1,2,2);
ax = gca;
ax.FontSize = 16;
ax.LineWidth = 1.2;

p1 = semilogy(time_egm,         rel_err_egm,         '--', 'Color', c4, 'LineWidth', 3); hold on;
p2 = semilogy(time_apdb_0,      rel_err_apdb_0,      '-',  'Color', c1, 'LineWidth', 3);
p3 = semilogy(time_rapdb_ada_0, rel_err_rapdb_ada_0, ':',  'Color', [0.466,0.674,0.188], 'LineWidth', 3);
p4 = semilogy(time_rapdb_0,     rel_err_rapdb_0,     ':',  'Color', [0.635,0.078,0.184], 'LineWidth', 3);

p5 = semilogy(time_apdb_1,      rel_err_apdb_1,      '--', 'Color', c1, ...
              'LineWidth', 3, 'Marker', 'o', 'MarkerSize', 9, ...
              'MarkerIndices', 1:gap_apdb_time:length(time_apdb_1));
p6 = semilogy(time_rapdb_ada_1, rel_err_rapdb_ada_1, '-',  'Color', [0.466,0.674,0.188], ...
              'LineWidth', 3, 'Marker', '^', 'MarkerSize', 9, ...
              'MarkerIndices', 1:gap_rapdbada_time:length(time_rapdb_ada_1));
p7 = semilogy(time_rapdb_1,     rel_err_rapdb_1,     '-',  'Color', [0.635,0.078,0.184], ...
              'LineWidth', 3, 'Marker', '+', 'MarkerSize', 9, ...
              'MarkerIndices', 1:gap_rapdb_time:length(time_rapdb_1));

grid on
hx = xlabel('Time (s)','Interpreter','latex');
hy = ylabel('$\|x^k-x^\star\|/(1+\|x^\star\|)$','Interpreter','latex');
hx.FontSize = 16;
hy.FontSize = 16;

lgd = legend([p1 p2 p3 p4 p5 p6 p7], ...
    {'EGM','APDB-yx (mono)','rAPDB-yx-ada (mono)','rAPDB-yx (mono)', ...
     'APDB-yx (nonmono)','rAPDB-yx-ada (nonmono)','rAPDB-yx (nonmono)'});
lgd.FontSize = 11;
lgd.Location = 'northeast';

%% -------- Save figure --------
saveas(fig, fullfile(outdir, 'comparison_plot.fig'));
saveas(fig, fullfile(outdir, 'comparison_plot.png'));
exportgraphics(fig, fullfile(outdir, 'comparison_plot.pdf'), 'ContentType', 'vector');

%% -------- Save experiment outputs --------
save(fullfile(outdir, 'experiment_outputs.mat'), ...
    'seed', 'data', 'nnn', 'reg_param', ...
    'optval', 'optsol', 'time_cvx', ...
    'rel_err_egm', 'time_egm', 'iter_egm', ...
    'rel_err_apdb_0', 'time_apdb_0', 'iter_apdb_0', ...
    'rel_err_rapdb_0', 'time_rapdb_0', 'iter_rapdb_0', ...
    'rel_err_rapdb_ada_0', 'time_rapdb_ada_0', 'iter_rapdb_ada_0', ...
    'rel_err_apdb_1', 'time_apdb_1', 'iter_apdb_1', ...
    'rel_err_rapdb_1', 'time_rapdb_1', 'iter_rapdb_1', ...
    'rel_err_rapdb_ada_1', 'time_rapdb_ada_1', 'iter_rapdb_ada_1', ...
    'ResultTable');

writetable(ResultTable, fullfile(outdir, 'summary_table.csv'));
save(fullfile(outdir, 'summary_table.mat'), 'ResultTable');

disp(' ')
disp(['All outputs have been saved to folder: ', outdir])