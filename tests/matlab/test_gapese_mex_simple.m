function Result = test_gapese_mex_simple()
%TEST_GAPESE_MEX_SIMPLE Compare C++/MEX Gapese with MATLAB Gapese.
%   The returned Result structure is also saved in the "results" folder
%   located next to this test file.

assert(exist('gapese_mex', 'file') == 3, ...
    'gapese_mex is not compiled. Run build_gapese_mex first.');
assert(exist('Gapese', 'class') == 8, ...
    'The reference MATLAB @Gapese class must be on the MATLAB path.');

rng(1);
Prob.n_x = 2;
Prob.n_lambda = 1;
Prob.n_b = 1;
Prob.n_d = 2;

Option = Gapese.create_Option();
Option.print_level = 0;
Option.D_gap_param_a = 0.5;
Option.D_gap_param_b = 2;
Option.tol_KKT_res = 1e-5;
Option.tol_D_gap_res = 1e-5;
Option.mu_max = 1e6;
Option.mu_min = 1e1;

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

Init.x = randn(Prob.n_x, 1);
Init.lambda = randn(Prob.n_lambda, 1);
Init.eta = randn(Prob.n_lambda, 1);

matlab_solver = Gapese(Prob, Option);
cpp_solver = GapeseMex(Prob, Option);

[Sol_matlab, Info_matlab] = matlab_solver.solve_CCQP(CCQP, Init);
[Sol_cpp, Info_cpp] = cpp_solver.solve_CCQP(CCQP, Init);

Result.solution.x_inf = norm(Sol_matlab.x - Sol_cpp.x, inf);
Result.solution.lambda_inf = norm(Sol_matlab.lambda - Sol_cpp.lambda, inf);
Result.solution.eta_inf = norm(Sol_matlab.eta - Sol_cpp.eta, inf);
Result.info.cost_abs = abs(Info_matlab.cost - Info_cpp.cost);
Result.info.dgap_abs = abs(Info_matlab.D_gap_res - Info_cpp.D_gap_res);
Result.info.kkt_abs = abs(Info_matlab.KKT_res - Info_cpp.KKT_res);
Result.Sol_matlab = Sol_matlab;
Result.Sol_cpp = Sol_cpp;
Result.Info_matlab = Info_matlab;
Result.Info_cpp = Info_cpp;

fprintf('\nMATLAB vs C++/MEX Gapese:\n');
fprintf('  status: MATLAB=%d, C++=%d\n', Info_matlab.status, Info_cpp.status);
fprintf('  iterations: MATLAB=%d, C++=%d\n', Info_matlab.iter_num, Info_cpp.iter_num);
fprintf('  ||dx||_inf      = %.3e\n', Result.solution.x_inf);
fprintf('  ||dlambda||_inf = %.3e\n', Result.solution.lambda_inf);
fprintf('  ||deta||_inf    = %.3e\n', Result.solution.eta_inf);
fprintf('  |dcost|         = %.3e\n', Result.info.cost_abs);
fprintf('  |dDgap|         = %.3e\n', Result.info.dgap_abs);
fprintf('  |dKKT|          = %.3e\n', Result.info.kkt_abs);

assert(Info_matlab.status == 1 && Info_cpp.status == 1, 'A solver failed.');
assert(Result.solution.x_inf < 1e-7, 'x solutions do not agree.');
assert(Result.solution.lambda_inf < 1e-7, 'lambda solutions do not agree.');
assert(Result.solution.eta_inf < 1e-7, 'eta solutions do not agree.');

% Reuse the same stateful C++ solver with a complete primal-dual warm start.
Init_warm.x = Sol_cpp.x;
Init_warm.lambda = Sol_cpp.lambda;
Init_warm.eta = Sol_cpp.eta;
Init_warm.v_d = Sol_cpp.v_d;
Init_warm.gamma_b = Sol_cpp.gamma_b;
Init_warm.gamma_d = Sol_cpp.gamma_d;
[Sol_warm, Info_warm] = cpp_solver.solve_CCQP(CCQP, Init_warm);
Result.Sol_warm = Sol_warm;
Result.Info_warm = Info_warm;
assert(Info_warm.status == 1, 'Stateful warm-start solve failed.');

fprintf('  stateful warm-start iterations = %d\n', Info_warm.iter_num);

%% Save test result next to this file.
test_dir = fileparts(mfilename('fullpath'));
result_dir = fullfile(test_dir, 'results');
if ~isfolder(result_dir)
    mkdir(result_dir);
end

Result.Timestamp = datestr(now, 30);
result_filename = sprintf('result_gapese_mex_simple_%s.mat', ...
    datestr(now, 'yyyymmdd_HHMMSS'));
result_path = fullfile(result_dir, result_filename);
Result.ResultFile = result_path;
save(result_path, 'Result', '-v7.3');

fprintf('\nSaved result file:\n  %s\n', result_path);
fprintf('All simple MEX tests passed.\n');
end
