function [best_tau, best_out, tune_table] = EGM_tune_tau(input,optval,x0,y0,epsilon,max_iter,opts)

if ~isfield(opts,'tau_grid')
    opts.tau_grid = logspace(-5,-2,10);
end
if ~isfield(opts,'tune_iter')
    opts.tune_iter = min(1000,max_iter);
end
if ~isfield(opts,'epoch')
    opts.epoch = 1;
end
if ~isfield(opts,'verbose')
    opts.verbose = 100;
end

tau_grid = opts.tau_grid;
num_tau = length(tau_grid);

score_list = inf(num_tau,1);
final_infeas = inf(num_tau,1);
final_subopt = inf(num_tau,1);

fprintf('\n========== Tuning EGM tau ==========\n');

for k = 1:num_tau
    tau = tau_grid(k);
    opts_tmp = opts;
    opts_tmp.tau = tau;

    try
        [rel_infeas, rel_subopt, ~, ~, ~] = ...
            EGM(input,optval,x0,y0,epsilon,opts.tune_iter,opts_tmp);

        if isempty(rel_infeas) || isempty(rel_subopt) ...
                || any(isnan(rel_infeas)) || any(isnan(rel_subopt)) ...
                || any(isinf(rel_infeas)) || any(isinf(rel_subopt))
            score = inf;
        else
            final_infeas(k) = rel_infeas(end);
            final_subopt(k) = rel_subopt(end);

            score = max(final_infeas(k), final_subopt(k));
        end

    catch ME
        fprintf('tau = %.2e failed: %s\n', tau, ME.message);
        score = inf;
    end

    score_list(k) = score;

    fprintf('tau = %.2e, score = %.3e, infeas = %.3e, subopt = %.3e\n', ...
        tau, score_list(k), final_infeas(k), final_subopt(k));
end

[~, idx] = min(score_list);
best_tau = tau_grid(idx);

fprintf('Best tau selected: %.2e\n', best_tau);
fprintf('====================================\n\n');

tune_table = table(tau_grid(:), score_list, final_infeas, final_subopt, ...
    'VariableNames', {'tau','score','final_infeas','final_subopt'});

opts_best = opts;
opts_best.tau = best_tau;

[rel_infeas_err,rel_subopt_err,time_period,iter_epoch,oracle] = ...
    EGM(input,optval,x0,y0,epsilon,max_iter,opts_best);

best_out.rel_infeas_err = rel_infeas_err;
best_out.rel_subopt_err = rel_subopt_err;
best_out.time_period = time_period;
best_out.iter_epoch = iter_epoch;
best_out.oracle = oracle;
best_out.opts = opts_best;

end