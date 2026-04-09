function [R, timing, x0] = blendenpik_matlab(A, sketch, gamma, isp, b)
% Blendenpik preconditioner: computes a sketch-based QR factorization of A.
% R is the upper triangular factor used as a preconditioner (A * R^{-1}).
% If isp=true, also solves the sketched system for an initial guess x0.
%
% Inputs:
%   A       - m x n matrix
%   sketch  - sketch type: 'SparseEmbedding' (default) or 'SRFT'
%   gamma   - sketch size multiplier, sketchsize = gamma*n (default: 3)
%   isp     - if true, compute initial solution x0 from sketched system (default: false)
%   b       - right-hand side vector (required if isp=true)

    if nargin < 2, sketch = 'SparseEmbedding'; end
    if nargin < 3, gamma  = 3;                 end
    if nargin < 4, isp    = false;             end


    [m, n]     = size(A);
    sketchsize = min(m, ceil(gamma * n));

    % ===== Sketching =====
    t0 = tic;
    switch sketch
        case 'SparseEmbedding'
            S  = sparsesign(sketchsize, m, 8);
            SA = S * A;
            if isp
                b = S * b;
            end
        case 'SRFT'
            SA = SRFT(A, sketchsize);
            if isp, b = SRFT(b, sketchsize); end
    end
    timing.sketch = toc(t0);

    % ===== QR factorization =====
    t1 = tic;
    if isp
        [Q, R] = qr(full(SA),0);
        x0 = R\(Q'*b); % initial solution from sketched least squares
    else
        R = qr(full(SA),0);
    end
    timing.qr = toc(t1);
end

function SA = SRFT(A,s)
% Subsampled Randomized Fourier Transform sketch
    m = size(A, 1);
    d = sign(randn(m, 1));
    if isreal(A)
        SA = dct(d .* full(A)); SA(1, :) = SA(1, :) / sqrt(2);
    else
        SA = fft(d.*A); % if A complex
    end
    ix = randsample(m,s);
    SA = SA(ix,:)*sqrt(m/s);
end
