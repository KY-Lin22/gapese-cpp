classdef GapeseMex < handle
    %GAPESEMEX Stateful MATLAB interface to the C++ Gapese solver.

    properties (SetAccess = private)
        Prob
        Option
    end

    properties (Access = private)
        Handle = uint64(0)
    end

    methods
        function self = GapeseMex(Prob, Option)
            if nargin < 2
                Option = GapeseMex.create_Option();
            end
            self.Prob = Prob;
            self.Option = Option;
            self.Handle = gapese_mex('create', Prob, Option);
        end

        function delete(self)
            if self.Handle ~= 0
                try
                    gapese_mex('destroy', self.Handle);
                catch
                    % MEX may already have been cleared during MATLAB shutdown.
                end
                self.Handle = uint64(0);
            end
        end

        function [Sol, Info] = solve_CCQP(self, CCQP, Init)
            if self.Handle == 0
                error('GapeseMex:InvalidHandle', 'The C++ solver handle has been destroyed.');
            end
            [Sol, Info] = gapese_mex('solve', self.Handle, CCQP, Init);
        end
    end

    methods (Static)
        function Option = create_Option()
            Option = struct();
            Option.print_level = 2;
            Option.D_gap_param_a = 0.1;
            Option.D_gap_param_b = 10;
            Option.FB_sigma = 0;
            Option.max_iter = 500;
            Option.tol_KKT_res = 1e-4;
            Option.tol_D_gap_res = 1e-4;
            Option.Newton_step_norm = 1e10;
            Option.mu_max = 1e6;
            Option.mu_min = 1e0;
            Option.mu_update = 2;
            Option.nu_x = 1e-8;
            Option.nu_xi = 1e-8;
            Option.nu_b = 1e-8;
            Option.nu_d = 1e-8;
            Option.beta_init = 1;
            Option.rho = 0.1;
            Option.alpha_min = 1e-16;
            Option.nu_alpha = 0.5;
            Option.nu_D = 1e-4;
        end
    end
end
