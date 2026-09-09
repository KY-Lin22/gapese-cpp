%SIMPLE_CCQP Minimal MATLAB MEX example for solving a CCQP with Gapese.

% Make the MATLAB wrapper and compiled MEX visible regardless of the
% caller's current working directory, then restore the original path.
this_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(fileparts(this_dir));
matlab_dir = fullfile(root_dir, 'matlab');
old_path = path;
path_cleanup = onCleanup(@() path(old_path)); %#ok<NASGU>
addpath(matlab_dir);

if exist('gapese_mex', 'file') ~= 3
    error('GapeseMex:MissingMex', ...
        'gapese_mex is not compiled. Run matlab/build_gapese_mex first.');
end

Prob.n_x = 2;
Prob.n_lambda = 1;
Prob.n_b = 1;
Prob.n_d = 2;

Option = GapeseMex.default_options();
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

fprintf('status: %d\n', Info.status);
fprintf('message: %s\n', Info.msg);
fprintf('x: [%g, %g]\n', Sol.x(1), Sol.x(2));
fprintf('lambda: %g\n', Sol.lambda);
fprintf('eta: %g\n', Sol.eta);
fprintf('KKT residual: %.6e\n', Info.KKT_res);
fprintf('D-gap residual: %.6e\n', Info.D_gap_res);
