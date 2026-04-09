function H = sketch_sparsesign_H(s, m, k)
% SKETCH_SPARSESIGN - Applies a sparse sign sketch to input matrix A.
%
% Syntax:
%   HA = sketch_sparsesign(s, m, k)
%
% Inputs:
%   m - original dimension
%   s - Target number of rows (sketch dimension).
%   k - Sparsity level (non-zero entries per column of the sketching matrix).
%
% Output:
%   H - sparse sign matrix of size s x m
%
% Note:
%   If the optimized MEX function 'sparsesign' is available, it is used.
%   Otherwise, it falls back to a slower MATLAB implementation and warns the user.

    if nargin < 3
        k = 8;
    end
    
    if exist('sparsesign','file') == 3
        H = sparsesign(s,m,k); % includes division by sqrt(k)
    else
        warning(['Using slow implementation sparsesign_slow.m. ' ...
            'For the better ' ...
            'implementation, build the mex file `sparsesign.c` ' ...
            'using the command `mex sparsesign.c`.']);
        H = sparsesign_slow(s,m,k);
    end
end