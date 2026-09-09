function build_gapese_mex(eigen_dir)
%BUILD_GAPESE_MEX Compile the stateful C++ Gapese MEX interface.
%   build_gapese_mex(EIGEN_DIR) expects EIGEN_DIR to contain the Eigen/
%   header directory. If omitted, EIGEN3_INCLUDE_DIR is read from the
%   environment, then <project>/third_party/eigen is tried.

this_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(this_dir);

if nargin < 1 || isempty(eigen_dir)
    eigen_dir = getenv('EIGEN3_INCLUDE_DIR');
else
    if isstring(eigen_dir)
        if ~isscalar(eigen_dir)
            error('GapeseMex:InvalidEigenPath', ...
                'EIGEN_DIR must be a scalar string or character vector.');
        end
        eigen_dir = char(eigen_dir);
    elseif ~ischar(eigen_dir)
        error('GapeseMex:InvalidEigenPath', ...
            'EIGEN_DIR must be a scalar string or character vector.');
    end
end
if isempty(eigen_dir)
    candidate = fullfile(root_dir, 'third_party', 'eigen');
    if isfolder(fullfile(candidate, 'Eigen'))
        eigen_dir = candidate;
    end
end
if isempty(eigen_dir) || ~isfolder(fullfile(eigen_dir, 'Eigen'))
    error('GapeseMex:EigenNotFound', [ ...
        'Eigen headers were not found. Call build_gapese_mex(<eigen-include-dir>) ', ...
        'or set the EIGEN3_INCLUDE_DIR environment variable.']);
end

sources = {
    fullfile(this_dir, 'gapese_mex.cpp')
    fullfile(root_dir, 'src', 'types.cpp')
    fullfile(root_dir, 'src', 'dgap.cpp')
    fullfile(root_dir, 'src', 'fb.cpp')
    fullfile(root_dir, 'src', 'kkt.cpp')
    fullfile(root_dir, 'src', 'merit.cpp')
    fullfile(root_dir, 'src', 'solver.cpp')
    };

include_root = ['-I', fullfile(root_dir, 'include')];
include_eigen = ['-I', eigen_dir];

if ispc
    compiler_flag = 'COMPFLAGS=$COMPFLAGS /std:c++17 /O2 /DNDEBUG';
else
    compiler_flag = 'CXXFLAGS=$CXXFLAGS -std=c++17 -O3 -DNDEBUG';
end

old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(this_dir);

mex('-R2018a', compiler_flag, include_root, include_eigen, ...
    sources{:}, '-output', 'gapese_mex');

fprintf('Built %s\n', fullfile(this_dir, ['gapese_mex.', mexext]));
end
