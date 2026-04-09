function [U, L, timing] = npcg(matvec, n, l0, lcap, q, tol, sketch)
% Adaptive Nyström approximation: U * L * U' ≈ matvec(eye(n)).
% Doubles sketch size each round until approximation error drops below tol.
%
% Inputs:
%   matvec  - function handle for matrix-vector product
%   n       - matrix dimension
%   l0      - initial sketch block size (doubles each round)
%   lcap    - maximum total sketch size
%   q       - number of power iterations for error estimation
%   tol     - target approximation error
%   sketch  - sketch type: 'sparse' (default) or 'gaussian'
%
% Outputs:
%   U, L    - Nyström factors: matvec ≈ U * diag(L) * U'
%   timing  - struct with timing breakdown

    if nargin < 7, sketch = 'sparse'; end

    Y = []; G = []; E = Inf;
    m = l0;

    timing_sketch1   = 0;
    timing_sketch2   = 0;
    timing_nystrom   = 0;
    timing_construct = 0;
    timing_errest    = 0;
    ttotal = tic;

    while E > tol

        % --- Draw sketch and apply matvec ---
        t0 = tic;
        G0 = draw_sketch(n, m, sketch);
        tsketch1 = tic;
        Y0 = matvec(G0);
        timing_sketch1 = timing_sketch1 + toc(tsketch1);

        G = [G, G0];
        Y = [Y, Y0];

        % --- Stabilized Nyström approximation ---
        [U, L, B, nu] = nystrom_factorize(G, Y);
        timing_nystrom = timing_nystrom + toc(t0);

        % --- SVD to extract factors ---
        t1 = tic;
        [U, L] = nystrom_svd(B, nu);
        timing_construct = timing_construct + toc(t1);

        % --- Error estimate via randomized power iteration ---
        t2 = tic;
        E = rand_power_err_est(matvec, U, L, q);
        timing_errest = timing_errest + toc(t2);

        % --- Double sketch size for next round ---
        m = l0; l0 = 2 * l0;

        % --- Handle rank cap: use remaining budget and break ---
        if l0 > lcap && E > tol
            m = lcap - (l0 - m);

            t0 = tic;
            G0 = draw_sketch(n, m, sketch);
            tsketch1 = tic;
            Y0 = matvec(G0);
            timing_sketch1 = timing_sketch1 + toc(tsketch1);

            G = [G, G0];
            Y = [Y, Y0];

            tsketch2 = tic;
            [U, L, B, nu] = nystrom_factorize(G, Y);
            timing_sketch2 = timing_sketch2 + toc(tsketch2);
            timing_nystrom = timing_nystrom + toc(t0);

            t1 = tic;
            [U, L] = nystrom_svd(B, nu);
            timing_construct = timing_construct + toc(t1);
            break;
        end
    end

    % Trim to actual rank
    r = size(B, 2);
    U = U(:, 1:r);
    L = L(1:r, 1:r);

    timing.sketch_time.x     = timing_sketch1;
    timing.sketch_time.y     = timing_sketch2;
    timing.sketch_time.total = timing_sketch1 + timing_sketch2;
    timing.nystrom            = timing_nystrom;
    timing.construct          = timing_construct;
    timing.errest             = timing_errest;
    timing.total              = toc(ttotal);
end


% =========================================================================
% Helper functions
% =========================================================================

function G0 = draw_sketch(n, m, sketch)
    % Draw a random sketch matrix of size n x m
    switch sketch
        case 'gaussian'
            G0 = randn(n, m);
        case 'sparse'
            G0 = sketch_sparsesign_H(m, n)';
    end
    if m > 0.95 * n
        [G0, ~] = qr(G0, 0);   % orthogonalize for near-square sketches
    end
end

function [U, L, B, nu] = nystrom_factorize(G, Y)
    % Stabilized Nyström core factorization
    nu   = eps(norm(Y, 'fro'));
    Ynu  = Y + nu * G;
    C    = chol(full(G' * Ynu));
    B    = Ynu / C;
    U    = B; L = B;   % placeholders; actual U, L extracted in nystrom_svd
end

function [U, L] = nystrom_svd(B, nu)
    % Extract Nyström factors from SVD of B, removing stability shift nu
    [U, S, ~] = svd(full(B), 0);
    L = diag(max(0, diag(S).^2 - nu));
end

function E = rand_power_err_est(matvec, U, L, q)
    % Estimate spectral norm of residual (matvec - U*L*U') via power iteration
    v = randn(size(U, 1), 1); v = v / norm(v);
    for i = 1:q
        w = matvec(v) - U * (L * (U' * v));
        E = v' * w;
        v = w / norm(w);
    end
end