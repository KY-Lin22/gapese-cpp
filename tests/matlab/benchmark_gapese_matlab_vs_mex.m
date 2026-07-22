function Results = benchmark_gapese_matlab_vs_mex(cfg)
%BENCHMARK_GAPESE_MATLAB_VS_MEX Compare MATLAB and C++/MEX Gapese.
%
%   Results = benchmark_gapese_matlab_vs_mex()
%   Results = benchmark_gapese_matlab_vs_mex(cfg)
%
% This benchmark:
%   1. Generates a reproducible sparse CCQP with a known feasible and
%      globally optimal complementary reference point.
%   2. Compares MATLAB Gapese and stateful C++/MEX Gapese numerically.
%   3. Measures solver construction time, solve wall time, and the internal
%      solve time reported by Info.Time.total.
%   4. Recomputes objective, feasibility, nonnegativity, and complementarity
%      metrics independently of either solver.
%
% Requirements:
%   - The pure MATLAB @Gapese class is on the MATLAB path.
%   - GapeseMex.m and the compiled gapese_mex MEX file are on the path.
%
% By default, result files are saved in the "results" folder located next
% to this benchmark file. A custom folder can be supplied through
% cfg.save_dir.
%
% Example:
%   Results = benchmark_gapese_matlab_vs_mex();
%
%   cfg = struct();
%   cfg.n_x = 250;
%   cfg.n_lambda = 150;
%   cfg.n_b = 80;
%   cfg.n_d = 350;
%   cfg.num_build_repeat = 20;
%   cfg.num_solve_repeat = 50;
%   Results = benchmark_gapese_matlab_vs_mex(cfg);

if nargin < 1 || isempty(cfg)
    cfg = struct();
end

% Use a results folder next to this test file, independent of MATLAB's
% current working directory.
test_dir = fileparts(mfilename('fullpath'));
default_save_dir = fullfile(test_dir, 'results');

cfg = local_complete_config(cfg, default_save_dir);
local_validate_environment();
local_validate_config(cfg);

fprintf('\n============================================================\n');
fprintf(' MATLAB Gapese vs C++/MEX Gapese benchmark\n');
fprintf('============================================================\n');
fprintf('Seed            : %d\n', cfg.seed);
fprintf('n_x             : %d\n', cfg.n_x);
fprintf('n_lambda        : %d\n', cfg.n_lambda);
fprintf('n_b             : %d\n', cfg.n_b);
fprintf('n_d             : %d\n', cfg.n_d);
fprintf('KKT dimension   : %d\n', ...
    cfg.n_x + 2*cfg.n_lambda + 2*cfg.n_d + cfg.n_b);
fprintf('Matrix density  : %.4f\n', cfg.matrix_density);
fprintf('Build repeats   : %d\n', cfg.num_build_repeat);
fprintf('Solve repeats   : %d\n', cfg.num_solve_repeat);

%% Generate one reproducible sparse CCQP.
rng(cfg.seed, 'twister');
[Prob, Option, CCQP, Init, Reference] = local_generate_problem(cfg);

fprintf('\nGenerated sparse CCQP:\n');
fprintf('  nnz(Q blocks) = %d\n', ...
    nnz(CCQP.Q_x) + nnz(CCQP.Q_lambda) + nnz(CCQP.Q_eta));
fprintf('  nnz(A blocks) = %d\n', ...
    nnz(CCQP.A_x) + nnz(CCQP.A_lambda) + nnz(CCQP.A_eta));
fprintf('  nnz(C blocks) = %d\n', ...
    nnz(CCQP.C_x) + nnz(CCQP.C_lambda) + nnz(CCQP.C_eta));

%% Constructor warm-up.
tmp_matlab = Gapese(Prob, Option);
clear tmp_matlab

tmp_mex = GapeseMex(Prob, Option);
delete(tmp_mex);
clear tmp_mex

%% Measure construction time.
build_matlab = zeros(cfg.num_build_repeat, 1);
build_mex = zeros(cfg.num_build_repeat, 1);
for i = 1:cfg.num_build_repeat
    t = tic;
    tmp_matlab = Gapese(Prob, Option); %#ok<NASGU>
    build_matlab(i) = toc(t);
    clear tmp_matlab

    t = tic;
    tmp_mex = GapeseMex(Prob, Option);
    build_mex(i) = toc(t);
    delete(tmp_mex);
    clear tmp_mex
end

%% Create persistent stateful solvers for numerical and solve-time tests.
solver_matlab = Gapese(Prob, Option);
solver_mex = GapeseMex(Prob, Option);
solver_cleanup = onCleanup(@() local_delete_if_valid(solver_mex)); %#ok<NASGU>

%% Warm up JIT, MEX loading, allocation, and sparse solver paths.
[~, ~] = solver_matlab.solve_CCQP(CCQP, Init);
[~, ~] = solver_mex.solve_CCQP(CCQP, Init);

%% One numerical regression solve.
[Sol_matlab, Info_matlab] = solver_matlab.solve_CCQP(CCQP, Init);
[Sol_mex, Info_mex] = solver_mex.solve_CCQP(CCQP, Init);

Metrics_matlab = local_solution_metrics(CCQP, Sol_matlab, Reference);
Metrics_mex = local_solution_metrics(CCQP, Sol_mex, Reference);
Numerical = local_compare_results(Sol_matlab, Info_matlab, ...
    Sol_mex, Info_mex, Metrics_matlab, Metrics_mex);

local_print_numerical(Numerical, Metrics_matlab, Metrics_mex, ...
    Info_matlab, Info_mex);

if cfg.assert_consistency
    assert(Info_matlab.status == 1, 'MATLAB Gapese failed on the benchmark problem.');
    assert(Info_mex.status == 1, 'C++/MEX Gapese failed on the benchmark problem.');
    assert(Numerical.x_inf <= cfg.solution_tolerance, ...
        'MATLAB and MEX x solutions differ beyond the configured tolerance.');
    assert(Numerical.lambda_inf <= cfg.solution_tolerance, ...
        'MATLAB and MEX lambda solutions differ beyond the configured tolerance.');
    assert(Numerical.eta_inf <= cfg.solution_tolerance, ...
        'MATLAB and MEX eta solutions differ beyond the configured tolerance.');
    assert(Numerical.cost_abs <= cfg.info_tolerance, ...
        'MATLAB and MEX objective values differ beyond the configured tolerance.');
    assert(Numerical.dgap_abs <= cfg.info_tolerance, ...
        'MATLAB and MEX D-gap residuals differ beyond the configured tolerance.');
    assert(Numerical.kkt_abs <= cfg.info_tolerance, ...
        'MATLAB and MEX KKT residuals differ beyond the configured tolerance.');
end

%% Repeated solve timing with identical initial guesses.
% Alternate solver order to reduce systematic drift from CPU load and cache.
wall_matlab = zeros(cfg.num_solve_repeat, 1);
wall_mex = zeros(cfg.num_solve_repeat, 1);
internal_matlab = zeros(cfg.num_solve_repeat, 1);
internal_mex = zeros(cfg.num_solve_repeat, 1);
iter_matlab = zeros(cfg.num_solve_repeat, 1);
iter_mex = zeros(cfg.num_solve_repeat, 1);
status_matlab = zeros(cfg.num_solve_repeat, 1);
status_mex = zeros(cfg.num_solve_repeat, 1);

for i = 1:cfg.num_solve_repeat
    if mod(i, 2) == 1
        [wall_matlab(i), internal_matlab(i), iter_matlab(i), status_matlab(i)] = ...
            local_timed_solve(solver_matlab, CCQP, Init);
        [wall_mex(i), internal_mex(i), iter_mex(i), status_mex(i)] = ...
            local_timed_solve(solver_mex, CCQP, Init);
    else
        [wall_mex(i), internal_mex(i), iter_mex(i), status_mex(i)] = ...
            local_timed_solve(solver_mex, CCQP, Init);
        [wall_matlab(i), internal_matlab(i), iter_matlab(i), status_matlab(i)] = ...
            local_timed_solve(solver_matlab, CCQP, Init);
    end
end

if any(status_matlab ~= 1)
    warning('benchmark:MatlabFailure', ...
        'MATLAB Gapese failed in %d timed repetitions.', nnz(status_matlab ~= 1));
end
if any(status_mex ~= 1)
    warning('benchmark:MexFailure', ...
        'C++/MEX Gapese failed in %d timed repetitions.', nnz(status_mex ~= 1));
end

Build = struct();
Build.MATLAB.samples = build_matlab;
Build.MEX.samples = build_mex;
Build.MATLAB.stats = local_stats(build_matlab);
Build.MEX.stats = local_stats(build_mex);

Solve = struct();
Solve.MATLAB.wall_samples = wall_matlab;
Solve.MEX.wall_samples = wall_mex;
Solve.MATLAB.internal_samples = internal_matlab;
Solve.MEX.internal_samples = internal_mex;
Solve.MATLAB.iter_samples = iter_matlab;
Solve.MEX.iter_samples = iter_mex;
Solve.MATLAB.status_samples = status_matlab;
Solve.MEX.status_samples = status_mex;
Solve.MATLAB.wall_stats = local_stats(wall_matlab);
Solve.MEX.wall_stats = local_stats(wall_mex);
Solve.MATLAB.internal_stats = local_stats(internal_matlab);
Solve.MEX.internal_stats = local_stats(internal_mex);
Solve.MATLAB.iter_stats = local_stats(iter_matlab);
Solve.MEX.iter_stats = local_stats(iter_mex);

Speedup = struct();
Speedup.build_mean = Build.MATLAB.stats.mean / Build.MEX.stats.mean;
Speedup.build_median = Build.MATLAB.stats.median / Build.MEX.stats.median;
Speedup.solve_wall_mean = Solve.MATLAB.wall_stats.mean / Solve.MEX.wall_stats.mean;
Speedup.solve_wall_median = Solve.MATLAB.wall_stats.median / Solve.MEX.wall_stats.median;
Speedup.solve_internal_mean = Solve.MATLAB.internal_stats.mean / Solve.MEX.internal_stats.mean;
Speedup.solve_internal_median = Solve.MATLAB.internal_stats.median / Solve.MEX.internal_stats.median;

local_print_timing(Build, Solve, Speedup);

%% Package results without storing solver objects.
Results = struct();
Results.Config = cfg;
Results.Problem = Prob;
Results.Option = Option;
Results.CCQP = CCQP;
Results.Init = Init;
Results.Reference = Reference;
Results.Solution.MATLAB = Sol_matlab;
Results.Solution.MEX = Sol_mex;
Results.Info.MATLAB = Info_matlab;
Results.Info.MEX = Info_mex;
Results.Metrics.MATLAB = Metrics_matlab;
Results.Metrics.MEX = Metrics_mex;
Results.Numerical = Numerical;
Results.Build = Build;
Results.Solve = Solve;
Results.Speedup = Speedup;
Results.Timestamp = datestr(now, 30);

if cfg.save_result
    cfg.save_dir = char(cfg.save_dir);
    if ~isfolder(cfg.save_dir)
        mkdir(cfg.save_dir);
    end
    filename = sprintf('result_gapese_matlab_vs_mex_%s.mat', ...
        datestr(now, 'yyyymmdd_HHMMSS'));
    result_path = fullfile(cfg.save_dir, filename);
    save(result_path, 'Results', '-v7.3');
    Results.ResultFile = result_path;
    fprintf('\nSaved result file:\n  %s\n', result_path);
else
    Results.ResultFile = '';
end

fprintf('\nBenchmark completed successfully.\n');
end

%% Configuration
function cfg = local_complete_config(cfg, default_save_dir)
defaults = struct();
defaults.seed = 1;
defaults.n_x = 120;
defaults.n_lambda = 80;
defaults.n_b = 40;
defaults.n_d = 180;
defaults.matrix_density = 0.05;
defaults.matrix_scale = 0.10;
defaults.init_noise = 0.10;
defaults.num_build_repeat = 20;
defaults.num_solve_repeat = 30;
defaults.solution_tolerance = 1e-6;
defaults.info_tolerance = 1e-6;
defaults.assert_consistency = true;
defaults.save_result = true;
defaults.save_dir = default_save_dir;

names = fieldnames(defaults);
for i = 1:numel(names)
    name = names{i};
    if ~isfield(cfg, name) || isempty(cfg.(name))
        cfg.(name) = defaults.(name);
    end
end
end

function local_validate_environment()
assert(exist('Gapese', 'class') == 8, ...
    'The pure MATLAB @Gapese class is not on the MATLAB path.');
assert(exist('GapeseMex', 'class') == 8, ...
    'GapeseMex.m is not on the MATLAB path.');
assert(exist('gapese_mex', 'file') == 3, ...
    'The compiled gapese_mex MEX file is not on the MATLAB path.');
end

function local_validate_config(cfg)
positive_integers = {'n_x', 'n_lambda', 'num_build_repeat', 'num_solve_repeat'};
for i = 1:numel(positive_integers)
    name = positive_integers{i};
    value = cfg.(name);
    assert(isscalar(value) && isfinite(value) && value > 0 && mod(value, 1) == 0, ...
        'cfg.%s must be a positive integer.', name);
end

nonnegative_integers = {'n_b', 'n_d'};
for i = 1:numel(nonnegative_integers)
    name = nonnegative_integers{i};
    value = cfg.(name);
    assert(isscalar(value) && isfinite(value) && value >= 0 && mod(value, 1) == 0, ...
        'cfg.%s must be a nonnegative integer.', name);
end

assert(cfg.n_b <= cfg.n_x + 2*cfg.n_lambda, ...
    'cfg.n_b cannot exceed the total number of CCQP variables.');
assert(isscalar(cfg.matrix_density) && cfg.matrix_density > 0 && cfg.matrix_density <= 1, ...
    'cfg.matrix_density must lie in (0,1].');
assert(isscalar(cfg.matrix_scale) && cfg.matrix_scale > 0, ...
    'cfg.matrix_scale must be positive.');
assert(isscalar(cfg.init_noise) && cfg.init_noise >= 0, ...
    'cfg.init_noise must be nonnegative.');
assert(isscalar(cfg.solution_tolerance) && cfg.solution_tolerance > 0, ...
    'cfg.solution_tolerance must be positive.');
assert(isscalar(cfg.info_tolerance) && cfg.info_tolerance > 0, ...
    'cfg.info_tolerance must be positive.');
assert(ischar(cfg.save_dir) || (isstring(cfg.save_dir) && isscalar(cfg.save_dir)), ...
    'cfg.save_dir must be a character vector or scalar string.');
end

%% Problem generation
function [Prob, Option, CCQP, Init, Reference] = local_generate_problem(cfg)
n_x = cfg.n_x;
n_lam = cfg.n_lambda;
n_b = cfg.n_b;
n_d = cfg.n_d;
n_total = n_x + 2*n_lam;

Prob = struct('n_x', n_x, 'n_lambda', n_lam, 'n_b', n_b, 'n_d', n_d);

Option = Gapese.create_Option();
Option.print_level = 0;
Option.D_gap_param_a = 0.5;
Option.D_gap_param_b = 2.0;
Option.tol_KKT_res = 1e-6;
Option.tol_D_gap_res = 1e-6;
Option.max_iter = 300;
Option.mu_min = 1e1;
Option.mu_max = 1e6;
Option.mu_update = 2;

% A strictly complementary reference point.
x_ref = 0.25*randn(n_x, 1);
mode_lambda = rand(n_lam, 1) >= 0.5;
lambda_ref = zeros(n_lam, 1);
eta_ref = zeros(n_lam, 1);
lambda_ref(mode_lambda) = 0.5 + rand(nnz(mode_lambda), 1);
eta_ref(~mode_lambda) = 0.5 + rand(nnz(~mode_lambda), 1);
z_ref = [x_ref; lambda_ref; eta_ref];

% Positive diagonal objective blocks centered at the reference point.
q_x = 1.0 + rand(n_x, 1);
q_lambda = 1.0 + rand(n_lam, 1);
q_eta = 1.0 + rand(n_lam, 1);
CCQP.Q_x = spdiags(q_x, 0, n_x, n_x);
CCQP.Q_lambda = spdiags(q_lambda, 0, n_lam, n_lam);
CCQP.Q_eta = spdiags(q_eta, 0, n_lam, n_lam);
CCQP.r_x = -q_x .* x_ref;
CCQP.r_lambda = -q_lambda .* lambda_ref;
CCQP.r_eta = -q_eta .* eta_ref;

% Equality matrix with an explicit pivot pattern to reduce rank problems.
if n_b > 0
    A_all = cfg.matrix_scale*sprandn(n_b, n_total, cfg.matrix_density);
    pivot_cols = 1:n_b;
    A_all = A_all + sparse(1:n_b, pivot_cols, ones(n_b, 1), n_b, n_total);
else
    A_all = sparse(0, n_total);
end
CCQP.A_x = A_all(:, 1:n_x);
CCQP.A_lambda = A_all(:, n_x + (1:n_lam));
CCQP.A_eta = A_all(:, n_x + n_lam + (1:n_lam));
CCQP.b = A_all*z_ref;

% Strictly feasible inequalities at the reference point.
if n_d > 0
    C_all = cfg.matrix_scale*sprandn(n_d, n_total, cfg.matrix_density);
    positive_slack = 0.5 + rand(n_d, 1);
else
    C_all = sparse(0, n_total);
    positive_slack = zeros(0, 1);
end
CCQP.C_x = C_all(:, 1:n_x);
CCQP.C_lambda = C_all(:, n_x + (1:n_lam));
CCQP.C_eta = C_all(:, n_x + n_lam + (1:n_lam));
CCQP.d = C_all*z_ref - positive_slack;

% Identical non-dual initial guess for every timed solve.
Init = struct();
Init.x = x_ref + cfg.init_noise*randn(n_x, 1);
Init.lambda = lambda_ref + cfg.init_noise*randn(n_lam, 1);
Init.eta = eta_ref + cfg.init_noise*randn(n_lam, 1);

Reference = struct();
Reference.x = x_ref;
Reference.lambda = lambda_ref;
Reference.eta = eta_ref;
Reference.z = z_ref;
Reference.inequality_slack = positive_slack;
Reference.objective = local_objective(CCQP, x_ref, lambda_ref, eta_ref);
end

%% Numerical checks
function Metrics = local_solution_metrics(CCQP, Sol, Reference)
equality = CCQP.A_x*Sol.x + CCQP.A_lambda*Sol.lambda + ...
    CCQP.A_eta*Sol.eta - CCQP.b;
inequality = CCQP.C_x*Sol.x + CCQP.C_lambda*Sol.lambda + ...
    CCQP.C_eta*Sol.eta - CCQP.d;

Metrics = struct();
Metrics.objective = local_objective(CCQP, Sol.x, Sol.lambda, Sol.eta);
Metrics.equality_inf = local_inf_norm(equality);
Metrics.inequality_violation_inf = local_inf_norm(min(inequality, 0));
Metrics.inequality_min_margin = local_min_or_inf(inequality);
Metrics.lambda_nonnegative_violation = local_inf_norm(min(Sol.lambda, 0));
Metrics.eta_nonnegative_violation = local_inf_norm(min(Sol.eta, 0));
Metrics.complementarity_product_inf = local_inf_norm(Sol.lambda .* Sol.eta);
Metrics.reference_distance_inf = local_inf_norm([...
    Sol.x - Reference.x;
    Sol.lambda - Reference.lambda;
    Sol.eta - Reference.eta]);
end

function Numerical = local_compare_results(Sol_matlab, Info_matlab, ...
    Sol_mex, Info_mex, Metrics_matlab, Metrics_mex)
Numerical = struct();
Numerical.x_inf = local_inf_norm(Sol_matlab.x - Sol_mex.x);
Numerical.lambda_inf = local_inf_norm(Sol_matlab.lambda - Sol_mex.lambda);
Numerical.eta_inf = local_inf_norm(Sol_matlab.eta - Sol_mex.eta);
Numerical.v_d_inf = local_inf_norm(Sol_matlab.v_d - Sol_mex.v_d);
Numerical.gamma_b_inf = local_inf_norm(Sol_matlab.gamma_b - Sol_mex.gamma_b);
Numerical.gamma_d_inf = local_inf_norm(Sol_matlab.gamma_d - Sol_mex.gamma_d);
Numerical.cost_abs = abs(Info_matlab.cost - Info_mex.cost);
Numerical.dgap_abs = abs(Info_matlab.D_gap_res - Info_mex.D_gap_res);
Numerical.kkt_abs = abs(Info_matlab.KKT_res - Info_mex.KKT_res);
Numerical.mu_abs = abs(Info_matlab.mu - Info_mex.mu);
Numerical.iter_difference = Info_matlab.iter_num - Info_mex.iter_num;
Numerical.status_equal = Info_matlab.status == Info_mex.status;
Numerical.independent_objective_abs = ...
    abs(Metrics_matlab.objective - Metrics_mex.objective);
end

function objective = local_objective(CCQP, x, lambda, eta)
objective = 0.5*(x.'*CCQP.Q_x*x) + CCQP.r_x.'*x ...
    + 0.5*(lambda.'*CCQP.Q_lambda*lambda) + CCQP.r_lambda.'*lambda ...
    + 0.5*(eta.'*CCQP.Q_eta*eta) + CCQP.r_eta.'*eta;
objective = full(objective);
end

%% Timing helpers
function [wall_time, internal_time, iter_num, status] = ...
    local_timed_solve(solver, CCQP, Init)
t = tic;
[~, Info] = solver.solve_CCQP(CCQP, Init);
wall_time = toc(t);
internal_time = Info.Time.total;
iter_num = Info.iter_num;
status = Info.status;
end

function stats = local_stats(values)
values = values(:);
stats = struct();
stats.mean = mean(values);
stats.median = median(values);
stats.minimum = min(values);
stats.maximum = max(values);
stats.std = std(values);
end

function local_delete_if_valid(object)
try
    if ~isempty(object) && isvalid(object)
        delete(object);
    end
catch
    % Ignore cleanup errors during MATLAB shutdown or MEX clearing.
end
end

%% Reporting
function local_print_numerical(N, Mmat, Mmex, Imat, Imex)
fprintf('\n------------------------------------------------------------\n');
fprintf('Numerical consistency\n');
fprintf('------------------------------------------------------------\n');
fprintf('Status                 : MATLAB=%d, MEX=%d\n', Imat.status, Imex.status);
fprintf('Iterations             : MATLAB=%d, MEX=%d\n', Imat.iter_num, Imex.iter_num);
fprintf('mu                     : MATLAB=%.3e, MEX=%.3e\n', Imat.mu, Imex.mu);
fprintf('||dx||_inf             : %.3e\n', N.x_inf);
fprintf('||dlambda||_inf        : %.3e\n', N.lambda_inf);
fprintf('||deta||_inf           : %.3e\n', N.eta_inf);
fprintf('||dv_d||_inf           : %.3e\n', N.v_d_inf);
fprintf('||dgamma_b||_inf       : %.3e\n', N.gamma_b_inf);
fprintf('||dgamma_d||_inf       : %.3e\n', N.gamma_d_inf);
fprintf('|dcost|                : %.3e\n', N.cost_abs);
fprintf('|dD-gap|               : %.3e\n', N.dgap_abs);
fprintf('|dKKT|                 : %.3e\n', N.kkt_abs);

fprintf('\nIndependent solution metrics:\n');
fprintf('                              MATLAB             MEX\n');
fprintf('  equality residual inf   %13.5e   %13.5e\n', ...
    Mmat.equality_inf, Mmex.equality_inf);
fprintf('  inequality violation    %13.5e   %13.5e\n', ...
    Mmat.inequality_violation_inf, Mmex.inequality_violation_inf);
fprintf('  minimum inequality gap  %13.5e   %13.5e\n', ...
    Mmat.inequality_min_margin, Mmex.inequality_min_margin);
fprintf('  lambda negative part    %13.5e   %13.5e\n', ...
    Mmat.lambda_nonnegative_violation, Mmex.lambda_nonnegative_violation);
fprintf('  eta negative part       %13.5e   %13.5e\n', ...
    Mmat.eta_nonnegative_violation, Mmex.eta_nonnegative_violation);
fprintf('  complementarity product %13.5e   %13.5e\n', ...
    Mmat.complementarity_product_inf, Mmex.complementarity_product_inf);
fprintf('  distance to reference   %13.5e   %13.5e\n', ...
    Mmat.reference_distance_inf, Mmex.reference_distance_inf);
end

function local_print_timing(Build, Solve, Speedup)
fprintf('\n------------------------------------------------------------\n');
fprintf('Timing results\n');
fprintf('------------------------------------------------------------\n');
fprintf('Metric                         MATLAB [s]       MEX [s]    Speedup\n');
fprintf('Constructor mean              %11.4e   %11.4e    %6.2fx\n', ...
    Build.MATLAB.stats.mean, Build.MEX.stats.mean, Speedup.build_mean);
fprintf('Constructor median            %11.4e   %11.4e    %6.2fx\n', ...
    Build.MATLAB.stats.median, Build.MEX.stats.median, Speedup.build_median);
fprintf('Solve wall mean               %11.4e   %11.4e    %6.2fx\n', ...
    Solve.MATLAB.wall_stats.mean, Solve.MEX.wall_stats.mean, ...
    Speedup.solve_wall_mean);
fprintf('Solve wall median             %11.4e   %11.4e    %6.2fx\n', ...
    Solve.MATLAB.wall_stats.median, Solve.MEX.wall_stats.median, ...
    Speedup.solve_wall_median);
fprintf('Solve internal mean           %11.4e   %11.4e    %6.2fx\n', ...
    Solve.MATLAB.internal_stats.mean, Solve.MEX.internal_stats.mean, ...
    Speedup.solve_internal_mean);
fprintf('Solve internal median         %11.4e   %11.4e    %6.2fx\n', ...
    Solve.MATLAB.internal_stats.median, Solve.MEX.internal_stats.median, ...
    Speedup.solve_internal_median);

fprintf('\nSolve wall-time spread:\n');
fprintf('  MATLAB: min=%.4e, max=%.4e, std=%.4e\n', ...
    Solve.MATLAB.wall_stats.minimum, Solve.MATLAB.wall_stats.maximum, ...
    Solve.MATLAB.wall_stats.std);
fprintf('  MEX   : min=%.4e, max=%.4e, std=%.4e\n', ...
    Solve.MEX.wall_stats.minimum, Solve.MEX.wall_stats.maximum, ...
    Solve.MEX.wall_stats.std);
fprintf('Mean iterations: MATLAB=%.2f, MEX=%.2f\n', ...
    Solve.MATLAB.iter_stats.mean, Solve.MEX.iter_stats.mean);
end

%% Small utility functions
function value = local_inf_norm(vector)
if isempty(vector)
    value = 0;
else
    value = norm(vector, inf);
end
end

function value = local_min_or_inf(vector)
if isempty(vector)
    value = inf;
else
    value = min(vector);
end
end
