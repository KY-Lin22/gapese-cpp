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
            if nargin < 2 || isempty(Option)
                Option = GapeseMex.default_options();
            else
                Option = GapeseMex.normalize_options(Option);
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

        function [Sol, Info] = solve_ccqp(self, CCQP, Init)
            if self.Handle == 0
                error('GapeseMex:InvalidHandle', 'The C++ solver handle has been destroyed.');
            end
            [Sol, Info] = gapese_mex('solve', self.Handle, CCQP, Init);
        end
    end

    methods (Static)
        function Option = default_options()
            %DEFAULT_OPTIONS Return the canonical Gapese option structure.
            Option = struct();
            Option.print_level = 2;
            Option.dgap_a = 0.1;
            Option.dgap_b = 10.0;
            Option.fb_sigma = 0.0;
            Option.max_iter = 500;
            Option.tol_kkt = 1e-5;
            Option.tol_dgap = 1e-6;
            Option.newton_step_max = 1e10;
            Option.mu_min = 1.0;
            Option.mu_max = 1e6;
            Option.mu_update = 2.0;
            Option.nu_x = 1e-8;
            Option.nu_xi = 1e-8;
            Option.nu_b = 1e-8;
            Option.nu_d = 1e-8;
            Option.beta_init = 1.0;
            Option.rho = 0.1;
            Option.alpha_min = 1e-16;
            Option.nu_alpha = 0.5;
            Option.nu_D = 1e-4;
        end
    end

    methods (Static, Access = private)
        function Option = normalize_options(Option)
            if ~isstruct(Option) || ~isscalar(Option)
                error('GapeseMex:InvalidOptions', ...
                    'Options must be provided as a scalar MATLAB struct.');
            end

            defaults = GapeseMex.default_options();
            names = fieldnames(Option);
            for i = 1:numel(names)
                name = names{i};
                if ~isfield(defaults, name)
                    error('GapeseMex:UnknownOption', ...
                        'Unknown Gapese option: %s', name);
                end
                defaults.(name) = Option.(name);
            end
            Option = defaults;
        end
    end
end
