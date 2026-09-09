function test_gapese_mex_simple()
%TEST_GAPESE_MEX_SIMPLE Lightweight self-contained test of the MATLAB MEX API.

% Make the MATLAB wrapper and compiled MEX visible regardless of the
% caller's current working directory, then restore the original path.
this_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(fileparts(this_dir));
matlab_dir = fullfile(root_dir, 'matlab');
old_path = path;
path_cleanup = onCleanup(@() path(old_path)); %#ok<NASGU>
addpath(matlab_dir);

assert(exist('gapese_mex', 'file') == 3, ...
    'gapese_mex is not compiled. Run build_gapese_mex first.');
assert(exist('GapeseMex', 'class') == 8, ...
    'GapeseMex.m must be on the MATLAB path.');

Prob.n_x = 2;
Prob.n_lambda = 1;
Prob.n_b = 1;
Prob.n_d = 2;

Option = GapeseMex.default_options();
expected_option_names = {
    'print_level'; 'dgap_a'; 'dgap_b'; 'fb_sigma'; 'max_iter'; ...
    'tol_kkt'; 'tol_dgap'; 'newton_step_max'; 'mu_min'; 'mu_max'; ...
    'mu_update'; 'nu_x'; 'nu_xi'; 'nu_b'; 'nu_d'; 'beta_init'; ...
    'rho'; 'alpha_min'; 'nu_alpha'; 'nu_D'};
assert(isequal(sort(fieldnames(Option)), sort(expected_option_names)), ...
    'default_options returned an unexpected option set.');
Option.print_level = 0;
Option.dgap_a = 0.5;
Option.dgap_b = 2.0;
Option.tol_kkt = 1e-5;
Option.tol_dgap = 1e-5;
Option.mu_min = 1e1;
Option.mu_max = 1e6;

CCQP.Q_x = sparse(diag([1, 1]));
CCQP.Q_lambda = sparse(1);
CCQP.Q_eta = sparse(1);
CCQP.r_x = [-1; -1];
CCQP.r_lambda = -1;
CCQP.r_eta = -1;
CCQP.A_x = sparse([1, 1]);
CCQP.A_lambda = sparse(Prob.n_b, Prob.n_lambda);
CCQP.A_eta = sparse(Prob.n_b, Prob.n_lambda);
CCQP.b = 1;
CCQP.C_x = speye(2);
CCQP.C_lambda = sparse(2, 1);
CCQP.C_eta = sparse(2, 1);
CCQP.d = zeros(2, 1);

Init.x = [0.25; -0.5];
Init.lambda = 0.3;
Init.eta = 0.7;

solver = GapeseMex(Prob, Option);
[Sol, Info] = solver.solve_ccqp(CCQP, Init);

assert(Info.status == 1, 'Gapese MEX solve failed: %s', Info.msg);
assert(Info.KKT_res <= 1.01 * Option.tol_kkt, ...
    'KKT residual exceeds the requested tolerance.');
assert(Info.D_gap_res <= 1.01 * Option.tol_dgap, ...
    'D-gap residual exceeds the requested tolerance.');
assert(abs(sum(Sol.x) - 1.0) <= 1e-7, ...
    'Equality constraint residual is too large.');
assert(all(Sol.x >= -1e-7), ...
    'Inequality constraint residual is too large.');

% Verify stateful repeated solve with a complete primal-dual warm start.
Init_warm.x = Sol.x;
Init_warm.lambda = Sol.lambda;
Init_warm.eta = Sol.eta;
Init_warm.v_d = Sol.v_d;
Init_warm.gamma_b = Sol.gamma_b;
Init_warm.gamma_d = Sol.gamma_d;
[Sol_warm, Info_warm] = solver.solve_ccqp(CCQP, Init_warm);
assert(Info_warm.status == 1, 'Stateful warm-start solve failed: %s', Info_warm.msg);
assert(norm(Sol_warm.x - Sol.x, inf) <= 1e-7, ...
    'Warm-started solution changed unexpectedly.');

% A partial option structure is allowed; unspecified fields use defaults.
partial = struct('print_level', 0, 'tol_kkt', 1e-5);
partial_solver = GapeseMex(Prob, partial);
assert(partial_solver.Option.tol_kkt == 1e-5, ...
    'Explicit partial option was not preserved.');
defaults = GapeseMex.default_options();
assert(partial_solver.Option.tol_dgap == defaults.tol_dgap, ...
    'Missing partial option did not fall back to the default value.');
clear partial_solver;

% Unknown names and removed legacy aliases must be rejected.
invalid_names = {
    'tol_kktt', ...
    'D_gap_param_a', ...
    'D_gap_param_b', ...
    'FB_sigma', ...
    'tol_KKT_res', ...
    'tol_D_gap_res', ...
    'Newton_step_norm'};
for i = 1:numel(invalid_names)
    bad = struct();
    bad.(invalid_names{i}) = 1;
    rejected = false;
    try
        GapeseMex(Prob, bad);
    catch ME
        rejected = strcmp(ME.identifier, 'GapeseMex:UnknownOption');
    end
    assert(rejected, 'Invalid option name was accepted: %s', invalid_names{i});
end

fprintf('All simple MATLAB MEX tests passed.\n');
end
