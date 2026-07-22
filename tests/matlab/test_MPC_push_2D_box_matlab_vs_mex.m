function Results = test_MPC_push_2D_box_matlab_vs_mex(cfg)
%TEST_MPC_PUSH_2D_BOX_MATLAB_VS_MEX Lockstep MPC comparison.
%
%   Results = test_MPC_push_2D_box_matlab_vs_mex()
%   Results = test_MPC_push_2D_box_matlab_vs_mex(cfg)
%
% This test compares the pure MATLAB Gapese solver and the stateful
% C++/MEX Gapese solver on the same online-linearized push-2D-box MPC
% problem.
%
% At each MPC step:
%   1. One shared nominal trajectory and one shared sparse CCQP are built.
%   2. MATLAB Gapese and GapeseMex solve exactly the same CCQP.
%   3. Their primal/dual solutions, first control, residuals and timings
%      are compared.
%   4. The shared nonlinear plant is propagated with the MATLAB control
%      after the paired controls have passed the consistency test.
%   5. Each solver keeps its own stateful primal-dual warm start.
%
% Default result directory:
%   <this file>/results
%
% Required MATLAB paths:
%   - Pure MATLAB gapese-matlab project, including OCP_push_2D_box.
%   - gapese-cpp/matlab, including GapeseMex and gapese_mex.
%
% Useful cfg fields:
%   cfg.simu_steps              default 600
%   cfg.T_samp                  default 0.03
%   cfg.use_dual_warm_start     default true
%   cfg.seed                    default 1
%   cfg.print_every             default 25
%   cfg.assert_consistency      default true
%   cfg.solution_tolerance      default 1e-6
%   cfg.info_tolerance          default 1e-6
%   cfg.save_result             default true
%   cfg.save_dir                default <this file>/results

if nargin < 1 || isempty(cfg)
    cfg = struct();
end

test_dir = fileparts(mfilename('fullpath'));
default_save_dir = fullfile(test_dir, 'results');
cfg = local_complete_config(cfg, default_save_dir);
local_validate_environment();
local_validate_config(cfg);

fprintf('\n============================================================\n');
fprintf(' Push-2D-box MPC: MATLAB Gapese vs C++/MEX Gapese\n');
fprintf('============================================================\n');
fprintf('Simulation steps        : %d\n', cfg.simu_steps);
fprintf('Sampling period         : %.4f s\n', cfg.T_samp);
fprintf('Dual warm start         : %d\n', cfg.use_dual_warm_start);
fprintf('Consistency assertions  : %d\n', cfg.assert_consistency);
fprintf('Plant propagation       : MATLAB control after lockstep comparison\n');

rng(cfg.seed, 'twister');

%% Create OCP data.
T = 0.3;
N = 10;

Data = struct();
Data.Q_x = sparse(diag([8e2; 8e2; 8e2]));
Data.Q_u = sparse(diag([0; 0; 5e0; 5e0; 5e0; 5e0]));
Data.Q_T = sparse(diag([1e0; 1e0; 1e0]));
Data.Q_du = sparse(diag([8e1; 8e1; 1e0; 1e0; 1e0; 1e0]));
Data.x_ref = [0; 0; 0];
Data.x_0 = [1; -1; 0];

alpha = linspace(0, 1, N + 1)';
Data.x_nominal = local_build_nominal_state( ...
    Data.x_0, Data.x_ref, alpha);
Data.u_nominal = randn(6, N);

OCP = OCP_push_2D_box(T, N, Data);
Prob = OCP.create_Gapese_Prob();

%% Solver options.
Option = Gapese.create_Option();
Option.D_gap_param_a = 0.1;
Option.D_gap_param_b = 10;
Option.tol_KKT_res = 1e-2;
Option.tol_D_gap_res = 1e-2;
Option.mu_max = 1e6;
Option.mu_min = 1e0;
Option.alpha_min = 1e-16;
Option.print_level = 0;

t_build = tic;
solver_matlab = Gapese(Prob, Option);
build_matlab = toc(t_build);

t_build = tic;
solver_mex = GapeseMex(Prob, Option);
build_mex = toc(t_build);
solver_cleanup = onCleanup(@() local_delete_if_valid(solver_mex)); %#ok<NASGU>

fprintf('MATLAB constructor time : %.4e s\n', build_matlab);
fprintf('MEX constructor time    : %.4e s\n', build_mex);

%% Shared plant and nominal data.
x_current = OCP.Data.x_0;
x_ref_current = OCP.Data.x_ref;
x_nominal_current = OCP.Data.x_nominal;
u_nominal_current = OCP.Data.u_nominal;

%% Offline cold solve using identical initial guesses.
Init_cold = struct();
Init_cold.x = randn(Prob.n_x, 1);
Init_cold.lambda = randn(Prob.n_lambda, 1);
Init_cold.eta = randn(Prob.n_lambda, 1);

DataUpdate = struct();
DataUpdate.x_0 = x_current;
DataUpdate.x_ref = x_ref_current;
DataUpdate.x_nominal = x_nominal_current;
DataUpdate.u_nominal = u_nominal_current;
CCQP = OCP.update_Gapese_CCQP(DataUpdate);

[Sol_cold_matlab, Info_cold_matlab, cold_wall_matlab] = ...
    local_timed_solve(solver_matlab, CCQP, Init_cold);
[Sol_cold_mex, Info_cold_mex, cold_wall_mex] = ...
    local_timed_solve(solver_mex, CCQP, Init_cold);

if Info_cold_matlab.status ~= 1 || Info_cold_mex.status ~= 1
    error('Offline cold solve failed: MATLAB=%d (%s), MEX=%d (%s).', ...
        Info_cold_matlab.status, Info_cold_matlab.msg, ...
        Info_cold_mex.status, Info_cold_mex.msg);
end

ColdDifference = local_solution_difference( ...
    Sol_cold_matlab, Info_cold_matlab, ...
    Sol_cold_mex, Info_cold_mex);

fprintf('\nOffline cold-start comparison:\n');
fprintf('  iterations: MATLAB=%d, MEX=%d\n', ...
    Info_cold_matlab.iter_num, Info_cold_mex.iter_num);
fprintf('  wall time : MATLAB=%.4e s, MEX=%.4e s\n', ...
    cold_wall_matlab, cold_wall_mex);
fprintf('  ||dx||_inf      = %.3e\n', ColdDifference.x_inf);
fprintf('  ||dlambda||_inf = %.3e\n', ColdDifference.lambda_inf);
fprintf('  ||deta||_inf    = %.3e\n', ColdDifference.eta_inf);

if cfg.assert_consistency
    local_assert_solver_pair( ...
        Sol_cold_matlab, Info_cold_matlab, ...
        Sol_cold_mex, Info_cold_mex, cfg, 0);
end

Init_matlab = local_make_warm_start( ...
    Sol_cold_matlab, OCP, cfg.use_dual_warm_start);
Init_mex = local_make_warm_start( ...
    Sol_cold_mex, OCP, cfg.use_dual_warm_start);

%% Allocate records.
Nsim = cfg.simu_steps;

Common = struct();
Common.x = [x_current, zeros(OCP.n_x, Nsim)];
Common.u = zeros(OCP.n_u, Nsim);
Common.x_ref = zeros(OCP.n_x, Nsim);
Common.CCQP_time = zeros(1, Nsim);
Common.simulation_time = zeros(1, Nsim);

Matlab = local_empty_solver_record(Nsim);
Mex = local_empty_solver_record(Nsim);

Difference = struct();
Difference.x_inf = zeros(1, Nsim);
Difference.lambda_inf = zeros(1, Nsim);
Difference.eta_inf = zeros(1, Nsim);
Difference.v_d_inf = zeros(1, Nsim);
Difference.gamma_b_inf = zeros(1, Nsim);
Difference.gamma_d_inf = zeros(1, Nsim);
Difference.control_inf = zeros(1, Nsim);
Difference.cost_abs = zeros(1, Nsim);
Difference.D_gap_abs = zeros(1, Nsim);
Difference.KKT_abs = zeros(1, Nsim);
Difference.mu_abs = zeros(1, Nsim);
Difference.iter_difference = zeros(1, Nsim);

executed_steps = 0;
MPC_status = 1;
failure_message = '';

%% Lockstep MPC simulation.
for k = 1:Nsim
    DataUpdate = struct();
    DataUpdate.x_0 = x_current;
    DataUpdate.x_ref = x_ref_current;
    DataUpdate.x_nominal = x_nominal_current;
    DataUpdate.u_nominal = u_nominal_current;

    t_ccqp = tic;
    CCQP = OCP.update_Gapese_CCQP(DataUpdate);
    Common.CCQP_time(k) = toc(t_ccqp);

    % Alternate solver order to reduce systematic timing bias.
    if mod(k, 2) == 1
        [Sol_matlab, Info_matlab, wall_matlab] = ...
            local_timed_solve(solver_matlab, CCQP, Init_matlab);
        [Sol_mex, Info_mex, wall_mex] = ...
            local_timed_solve(solver_mex, CCQP, Init_mex);
    else
        [Sol_mex, Info_mex, wall_mex] = ...
            local_timed_solve(solver_mex, CCQP, Init_mex);
        [Sol_matlab, Info_matlab, wall_matlab] = ...
            local_timed_solve(solver_matlab, CCQP, Init_matlab);
    end

    Matlab = local_set_solver_step( ...
        Matlab, k, Info_matlab, wall_matlab);
    Mex = local_set_solver_step( ...
        Mex, k, Info_mex, wall_mex);

    if Info_matlab.status ~= 1 || Info_mex.status ~= 1
        MPC_status = 0;
        failure_message = sprintf( ...
            'MPC step %d solver failure: MATLAB=%d (%s), MEX=%d (%s).', ...
            k, Info_matlab.status, Info_matlab.msg, ...
            Info_mex.status, Info_mex.msg);
        warning('Push2DBox:SolverFailure', '%s', failure_message);
        break;
    end

    Diff_k = local_solution_difference( ...
        Sol_matlab, Info_matlab, Sol_mex, Info_mex);

    Difference.x_inf(k) = Diff_k.x_inf;
    Difference.lambda_inf(k) = Diff_k.lambda_inf;
    Difference.eta_inf(k) = Diff_k.eta_inf;
    Difference.v_d_inf(k) = Diff_k.v_d_inf;
    Difference.gamma_b_inf(k) = Diff_k.gamma_b_inf;
    Difference.gamma_d_inf(k) = Diff_k.gamma_d_inf;
    Difference.cost_abs(k) = Diff_k.cost_abs;
    Difference.D_gap_abs(k) = Diff_k.D_gap_abs;
    Difference.KKT_abs(k) = Diff_k.KKT_abs;
    Difference.mu_abs(k) = Diff_k.mu_abs;
    Difference.iter_difference(k) = Diff_k.iter_difference;

    % Recover absolute trajectories from the two increment solutions.
    DXU_matlab = reshape( ...
        Sol_matlab.x, OCP.n_x + OCP.n_u, OCP.N);
    DXU_mex = reshape( ...
        Sol_mex.x, OCP.n_x + OCP.n_u, OCP.N);

    U_matlab = OCP.Data.u_nominal + ...
        DXU_matlab(OCP.n_x + 1:end, :);
    U_mex = OCP.Data.u_nominal + ...
        DXU_mex(OCP.n_x + 1:end, :);

    u_matlab = U_matlab(:, 1);
    u_mex = U_mex(:, 1);
    Difference.control_inf(k) = norm(u_matlab - u_mex, inf);

    if cfg.assert_consistency
        local_assert_solver_pair( ...
            Sol_matlab, Info_matlab, ...
            Sol_mex, Info_mex, cfg, k);
        assert(Difference.control_inf(k) <= cfg.solution_tolerance, ...
            'MPC step %d control inputs differ beyond tolerance.', k);
    end

    % Propagate one shared nonlinear plant with the MATLAB control.
    t_sim = tic;
    x_next = OCP.simulate_2D_box( ...
        x_current, u_matlab, cfg.T_samp);
    Common.simulation_time(k) = toc(t_sim);

    if any(~isfinite(x_next))
        MPC_status = 0;
        failure_message = sprintf( ...
            'MPC step %d nonlinear simulation returned non-finite values.', k);
        warning('Push2DBox:SimulationFailure', '%s', failure_message);
        break;
    end

    executed_steps = k;
    Common.x(:, k + 1) = x_next;
    Common.u(:, k) = u_matlab;
    Common.x_ref(:, k) = x_ref_current;

    if cfg.print_every > 0 && ...
            (k == 1 || mod(k, cfg.print_every) == 0 || k == Nsim)
        fprintf(['step %4d/%4d | iter M/MEX %3d/%3d | ', ...
            'solve %.3e/%.3e s | du %.2e | dx %.2e\n'], ...
            k, Nsim, Info_matlab.iter_num, Info_mex.iter_num, ...
            wall_matlab, wall_mex, ...
            Difference.control_inf(k), Difference.x_inf(k));
    end

    % Shared measured state.
    x_current = x_next;

    % Preserve the original reference/nominal update timing. A change
    % triggered at step k is first used in step k+1.
    switch k
        case 200
            x_ref_current = [1; 1; 0];
            u_nominal_current = randn(OCP.n_u, OCP.N);
        case 400
            x_ref_current = [1; 0; 0];
            u_nominal_current = randn(OCP.n_u, OCP.N);
        otherwise
            u_nominal_current = repmat( ...
                u_matlab, 1, OCP.N);
    end
    x_nominal_current = local_build_nominal_state( ...
        x_current, x_ref_current, alpha);

    % Each solver maintains its own horizon-shifted primal-dual warm start.
    Init_matlab = local_make_warm_start( ...
        Sol_matlab, OCP, cfg.use_dual_warm_start);
    Init_mex = local_make_warm_start( ...
        Sol_mex, OCP, cfg.use_dual_warm_start);
end

%% Truncate records and summarize.
Common = local_trim_common(Common, executed_steps);
Matlab = local_trim_solver_record(Matlab, executed_steps);
Mex = local_trim_solver_record(Mex, executed_steps);
Difference = local_trim_difference(Difference, executed_steps);

Summary = local_make_summary( ...
    Common, Matlab, Mex, Difference, ...
    build_matlab, build_mex, executed_steps);
local_print_summary(Summary, MPC_status, failure_message, cfg);

%% Package plain MATLAB data.
Results = struct();
Results.Config = cfg;
Results.Problem = Prob;
Results.Option = Option;
Results.Status = MPC_status;
Results.FailureMessage = failure_message;
Results.ExecutedSteps = executed_steps;
Results.Constructor.MATLAB = build_matlab;
Results.Constructor.MEX = build_mex;

Results.Offline.MATLAB.Solution = Sol_cold_matlab;
Results.Offline.MATLAB.Info = Info_cold_matlab;
Results.Offline.MATLAB.WallTime = cold_wall_matlab;
Results.Offline.MEX.Solution = Sol_cold_mex;
Results.Offline.MEX.Info = Info_cold_mex;
Results.Offline.MEX.WallTime = cold_wall_mex;
Results.Offline.Difference = ColdDifference;

Results.Common = Common;
Results.MATLAB = Matlab;
Results.MEX = Mex;
Results.Difference = Difference;
Results.Summary = Summary;
Results.Timestamp = datestr(now, 30);

if cfg.save_result
    save_dir = char(cfg.save_dir);
    if ~isfolder(save_dir)
        mkdir(save_dir);
    end

    filename = sprintf( ...
        'result_MPC_push_2D_box_matlab_vs_mex_%s.mat', ...
        datestr(now, 'yyyymmdd_HHMMSS'));
    result_path = fullfile(save_dir, filename);
    Results.ResultFile = result_path;
    save(result_path, 'Results', '-v7.3');

    fprintf('\nSaved result file:\n  %s\n', result_path);
else
    Results.ResultFile = '';
end

fprintf('\nPush-2D-box MATLAB/MEX comparison completed.\n');
end

%% Configuration
function cfg = local_complete_config(cfg, default_save_dir)
defaults = struct();
defaults.T_samp = 0.03;
defaults.simu_steps = 600;
defaults.use_dual_warm_start = true;
defaults.seed = 1;
defaults.print_every = 25;
defaults.assert_consistency = true;
defaults.solution_tolerance = 1e-6;
defaults.info_tolerance = 1e-6;
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
    'The pure MATLAB @Gapese class is not on the path.');
assert(exist('GapeseMex', 'class') == 8, ...
    'GapeseMex.m is not on the path.');
assert(exist('gapese_mex', 'file') == 3, ...
    'The compiled gapese_mex MEX file is not on the path.');
assert(exist('OCP_push_2D_box', 'class') == 8, ...
    'The OCP_push_2D_box class is not on the path.');
end

function local_validate_config(cfg)
assert(isscalar(cfg.T_samp) && isnumeric(cfg.T_samp) && ...
    isfinite(cfg.T_samp) && cfg.T_samp > 0, ...
    'cfg.T_samp must be a positive finite scalar.');
assert(isscalar(cfg.simu_steps) && isnumeric(cfg.simu_steps) && ...
    isfinite(cfg.simu_steps) && cfg.simu_steps > 0 && ...
    mod(cfg.simu_steps, 1) == 0, ...
    'cfg.simu_steps must be a positive integer.');
assert(isscalar(cfg.print_every) && isnumeric(cfg.print_every) && ...
    isfinite(cfg.print_every) && cfg.print_every >= 0 && ...
    mod(cfg.print_every, 1) == 0, ...
    'cfg.print_every must be a nonnegative integer.');
assert(isscalar(cfg.solution_tolerance) && ...
    cfg.solution_tolerance > 0, ...
    'cfg.solution_tolerance must be positive.');
assert(isscalar(cfg.info_tolerance) && ...
    cfg.info_tolerance > 0, ...
    'cfg.info_tolerance must be positive.');
assert(ischar(cfg.save_dir) || ...
    (isstring(cfg.save_dir) && isscalar(cfg.save_dir)), ...
    'cfg.save_dir must be a character vector or scalar string.');
end

%% Solver helpers
function [Sol, Info, wall_time] = local_timed_solve(solver, CCQP, Init)
t = tic;
[Sol, Info] = solver.solve_CCQP(CCQP, Init);
wall_time = toc(t);
end

function Init = local_make_warm_start(Sol, OCP, use_dual)
DXU = reshape(Sol.x, OCP.n_x + OCP.n_u, OCP.N);
Lambda = reshape(Sol.lambda, OCP.n_c, OCP.N);
Eta = reshape(Sol.eta, OCP.n_c, OCP.N);

Init = struct();
Init.x = reshape([DXU(:, 2:end), DXU(:, end)], [], 1);
Init.lambda = reshape([Lambda(:, 2:end), Lambda(:, end)], [], 1);
Init.eta = reshape([Eta(:, 2:end), Eta(:, end)], [], 1);

if use_dual
    Init.v_d = Sol.v_d;
    Init.gamma_b = Sol.gamma_b;
    Init.gamma_d = Sol.gamma_d;
end
end

function Diff = local_solution_difference(Sm, Im, Sx, Ix)
Diff = struct();
Diff.x_inf = local_inf_norm(Sm.x - Sx.x);
Diff.lambda_inf = local_inf_norm(Sm.lambda - Sx.lambda);
Diff.eta_inf = local_inf_norm(Sm.eta - Sx.eta);
Diff.v_d_inf = local_inf_norm(Sm.v_d - Sx.v_d);
Diff.gamma_b_inf = local_inf_norm(Sm.gamma_b - Sx.gamma_b);
Diff.gamma_d_inf = local_inf_norm(Sm.gamma_d - Sx.gamma_d);
Diff.cost_abs = abs(Im.cost - Ix.cost);
Diff.D_gap_abs = abs(Im.D_gap_res - Ix.D_gap_res);
Diff.KKT_abs = abs(Im.KKT_res - Ix.KKT_res);
Diff.mu_abs = abs(Im.mu - Ix.mu);
Diff.iter_difference = Im.iter_num - Ix.iter_num;
end

function local_assert_solver_pair(Sm, Im, Sx, Ix, cfg, step)
prefix = sprintf('MPC step %d', step);

assert(Im.status == 1 && Ix.status == 1, ...
    '%s: one of the solvers failed.', prefix);
assert(local_inf_norm(Sm.x - Sx.x) <= cfg.solution_tolerance, ...
    '%s: x solutions differ beyond tolerance.', prefix);
assert(local_inf_norm(Sm.lambda - Sx.lambda) <= cfg.solution_tolerance, ...
    '%s: lambda solutions differ beyond tolerance.', prefix);
assert(local_inf_norm(Sm.eta - Sx.eta) <= cfg.solution_tolerance, ...
    '%s: eta solutions differ beyond tolerance.', prefix);
assert(abs(Im.cost - Ix.cost) <= cfg.info_tolerance, ...
    '%s: costs differ beyond tolerance.', prefix);
assert(abs(Im.D_gap_res - Ix.D_gap_res) <= cfg.info_tolerance, ...
    '%s: D-gap residuals differ beyond tolerance.', prefix);
assert(abs(Im.KKT_res - Ix.KKT_res) <= cfg.info_tolerance, ...
    '%s: KKT residuals differ beyond tolerance.', prefix);
end

function Record = local_empty_solver_record(N)
Record = struct();
Record.solve_wall = zeros(1, N);
Record.solve_internal = zeros(1, N);
Record.iter_num = zeros(1, N);
Record.status = zeros(1, N);
Record.D_gap_res = zeros(1, N);
Record.KKT_res = zeros(1, N);
Record.mu = zeros(1, N);
Record.cost = zeros(1, N);
Record.time_check_termination = zeros(1, N);
Record.time_search_direction = zeros(1, N);
Record.time_step_size = zeros(1, N);
Record.time_else = zeros(1, N);
end

function Record = local_set_solver_step(Record, n, Info, wall_time)
Record.solve_wall(n) = wall_time;
Record.solve_internal(n) = Info.Time.total;
Record.iter_num(n) = Info.iter_num;
Record.status(n) = Info.status;
Record.D_gap_res(n) = Info.D_gap_res;
Record.KKT_res(n) = Info.KKT_res;
Record.mu(n) = Info.mu;
Record.cost(n) = Info.cost;
Record.time_check_termination(n) = Info.Time.check_termination;
Record.time_search_direction(n) = Info.Time.search_direction;
Record.time_step_size(n) = Info.Time.step_size;
Record.time_else(n) = Info.Time.else;
end

%% Record trimming and summaries
function Common = local_trim_common(Common, n)
Common.x = Common.x(:, 1:n + 1);
Common.u = Common.u(:, 1:n);
Common.x_ref = Common.x_ref(:, 1:n);
Common.CCQP_time = Common.CCQP_time(1:n);
Common.simulation_time = Common.simulation_time(1:n);
end

function Record = local_trim_solver_record(Record, n)
names = fieldnames(Record);
for i = 1:numel(names)
    Record.(names{i}) = Record.(names{i})(1:n);
end
end

function Difference = local_trim_difference(Difference, n)
names = fieldnames(Difference);
for i = 1:numel(names)
    Difference.(names{i}) = Difference.(names{i})(1:n);
end
end

function Summary = local_make_summary( ...
    Common, M, X, D, build_m, build_x, n)

Summary = struct();
Summary.executed_steps = n;
Summary.constructor_MATLAB = build_m;
Summary.constructor_MEX = build_x;

if n == 0
    Summary.mean_CCQP_time = NaN;
    Summary.mean_simulation_time = NaN;
    Summary.MATLAB = local_nan_solver_summary();
    Summary.MEX = local_nan_solver_summary();
    Summary.speedup_wall = NaN;
    Summary.speedup_internal = NaN;
    Summary.max_difference = struct();
    return;
end

Summary.mean_CCQP_time = mean(Common.CCQP_time);
Summary.mean_simulation_time = mean(Common.simulation_time);
Summary.MATLAB = local_solver_summary(M, Common);
Summary.MEX = local_solver_summary(X, Common);

Summary.speedup_wall = ...
    Summary.MATLAB.mean_solve_wall / ...
    Summary.MEX.mean_solve_wall;
Summary.speedup_internal = ...
    Summary.MATLAB.mean_solve_internal / ...
    Summary.MEX.mean_solve_internal;

Summary.max_difference = struct();
names = fieldnames(D);
for i = 1:numel(names)
    values = D.(names{i});
    if isempty(values)
        Summary.max_difference.(names{i}) = NaN;
    else
        Summary.max_difference.(names{i}) = max(abs(values));
    end
end
end

function S = local_solver_summary(R, Common)
S = struct();
S.mean_solve_wall = mean(R.solve_wall);
S.median_solve_wall = median(R.solve_wall);
S.max_solve_wall = max(R.solve_wall);
S.mean_solve_internal = mean(R.solve_internal);
S.median_solve_internal = median(R.solve_internal);
S.max_solve_internal = max(R.solve_internal);
S.mean_iterations = mean(R.iter_num);
S.max_iterations = max(R.iter_num);
S.failure_count = nnz(R.status ~= 1);
S.mean_total_step_equivalent = mean( ...
    Common.CCQP_time + ...
    R.solve_wall + ...
    Common.simulation_time);
end

function S = local_nan_solver_summary()
S = struct();
S.mean_solve_wall = NaN;
S.median_solve_wall = NaN;
S.max_solve_wall = NaN;
S.mean_solve_internal = NaN;
S.median_solve_internal = NaN;
S.max_solve_internal = NaN;
S.mean_iterations = NaN;
S.max_iterations = NaN;
S.failure_count = NaN;
S.mean_total_step_equivalent = NaN;
end

function local_print_summary(S, status, failure_message, cfg)
fprintf('\n------------------------------------------------------------\n');
fprintf('Push-2D-box MPC comparison summary\n');
fprintf('------------------------------------------------------------\n');
fprintf('Status                         : %d\n', status);
fprintf('Executed steps                 : %d / %d\n', ...
    S.executed_steps, cfg.simu_steps);

if ~isempty(failure_message)
    fprintf('Failure message                : %s\n', ...
        failure_message);
end

if S.executed_steps == 0
    return;
end

fprintf('Mean CCQP transcription        : %.4e s\n', ...
    S.mean_CCQP_time);
fprintf('Mean nonlinear simulation      : %.4e s\n', ...
    S.mean_simulation_time);
fprintf('\n');
fprintf('Metric                            MATLAB          MEX\n');
fprintf('Mean solve wall [s]           %12.4e %12.4e\n', ...
    S.MATLAB.mean_solve_wall, S.MEX.mean_solve_wall);
fprintf('Median solve wall [s]         %12.4e %12.4e\n', ...
    S.MATLAB.median_solve_wall, S.MEX.median_solve_wall);
fprintf('Max solve wall [s]            %12.4e %12.4e\n', ...
    S.MATLAB.max_solve_wall, S.MEX.max_solve_wall);
fprintf('Mean solve internal [s]       %12.4e %12.4e\n', ...
    S.MATLAB.mean_solve_internal, ...
    S.MEX.mean_solve_internal);
fprintf('Mean iterations               %12.2f %12.2f\n', ...
    S.MATLAB.mean_iterations, S.MEX.mean_iterations);
fprintf('Max iterations                %12.0f %12.0f\n', ...
    S.MATLAB.max_iterations, S.MEX.max_iterations);
fprintf('Mean equivalent MPC step [s]  %12.4e %12.4e\n', ...
    S.MATLAB.mean_total_step_equivalent, ...
    S.MEX.mean_total_step_equivalent);

fprintf('\nMATLAB/MEX speedup: wall=%.3fx, internal=%.3fx\n', ...
    S.speedup_wall, S.speedup_internal);
fprintf('Maximum paired differences:\n');
fprintf('  ||dx||_inf      = %.3e\n', ...
    S.max_difference.x_inf);
fprintf('  ||dlambda||_inf = %.3e\n', ...
    S.max_difference.lambda_inf);
fprintf('  ||deta||_inf    = %.3e\n', ...
    S.max_difference.eta_inf);
fprintf('  ||du||_inf      = %.3e\n', ...
    S.max_difference.control_inf);
fprintf('  |dcost|         = %.3e\n', ...
    S.max_difference.cost_abs);
fprintf('  |dD-gap|        = %.3e\n', ...
    S.max_difference.D_gap_abs);
fprintf('  |dKKT|          = %.3e\n', ...
    S.max_difference.KKT_abs);
end

%% Utilities
function X_nom = local_build_nominal_state(x0, xref, alpha)
X_nom = (1 - alpha') .* x0 + alpha' .* xref;
end

function local_delete_if_valid(object)
try
    if ~isempty(object) && isvalid(object)
        delete(object);
    end
catch
    % Ignore cleanup errors during shutdown or MEX clearing.
end
end

function value = local_inf_norm(vector)
if isempty(vector)
    value = 0;
else
    value = norm(vector, inf);
end
end
