function [xs, ts_xs, resvecs, ts_resvecs, timings, timestamps, ls, flag, matvec, pre] = ...
    aplicur(A, mu, Amu, bmu, block_size, err_tol_, max_it, params_lsqr, verbose, ns_var)

    % ===== Defaults for optional fields in ns_var =====
    if ~isfield(ns_var, 'sketch'),      ns_var.sketch      = 'SparseEmbedding'; end
    if ~isfield(ns_var, 'seed'),        ns_var.seed        = 11;                end
    if ~isfield(ns_var, 'tol_errdist'), ns_var.tol_errdist = 10;               end
    if ~isfield(ns_var, 'tol_cvgrate'), ns_var.tol_cvgrate = 100;              end
    if ~isfield(ns_var, 'svdfree'),     ns_var.svdfree     = false;            end
    rng(ns_var.seed);

    % Handle regularized vs non-regularized case
    if ns_var.svdfree && mu
        % SVD-free variant sketches Amu directly to avoid SVD of original A
        A = Amu;
        err_tol_ = 5 * err_tol_;   % relax tolerance for augmented system
    else
        if ~mu, Amu = A; bmu = bmu(1:size(A,1)); end
    end

    params_lsqr.tol_cvgrate = ns_var.tol_cvgrate;
    fprintf("svdfree=%d, tol_errdist=%d, tol_cvgrate=%d\n", ...
        ns_var.svdfree, ns_var.tol_errdist, ns_var.tol_cvgrate);

    % ===== Problem setup =====
    [rows, cols] = size(A);
    block_size   = min(block_size, min(rows, cols));
    err_tol      = err_tol_;
    err_dist     = Inf;

    apply   = @(x) Amu * x;
    applyt  = @(x) Amu' * x;
    summary = @(mv) @(x) norm(mv(x) - bmu);

    final_max_it = params_lsqr.maxit;
    inter_max_it = max(5, round(cols / 500));

    timings.timing_approx      = 0;
    timings.timing_aplicurprec = 0;
    timings.timing_aplicurlsqr = 0;

    % ===== Preallocation =====
    max_reslen = ceil(log2(max_it)) * inter_max_it + final_max_it;
    resvecs    = zeros(max_reslen, 1);
    ts_resvecs = zeros(max_reslen + ceil(log2(max_it)), 1);
    xs         = zeros(cols, max_reslen);
    ts_xs      = zeros(cols, max_reslen + ceil(log2(max_it)));

    % ===== Sketching =====
    t0 = tic;
    disp(ns_var.sketch);
    tsketch = tic;
    switch ns_var.sketch
        case 'Gaussian'
            S  = randn(ceil(1.1 * block_size), rows);
            SA = S * A;
        case 'SRFT'
            SA = SRFT(A, ceil(1.1 * block_size));
        case 'SparseEmbedding'
            S  = sparsesign(ceil(1.1 * block_size), rows, 8);
            SA = S * A;
    end
    anorm   = estimated_2norm(SA);
    amunorm = sqrt(anorm^2 + mu^2);
    timings.sketch = toc(tsketch);

    % ===== CUR initialization =====
    ResL = SA;
    I = zeros(block_size * max_it, 1);
    J = zeros(block_size * max_it, 1);
    C = A; Q = A; T = A; R = A; R_new = [];
    start_idx_row = 1; start_idx_col = 1;
    errors = zeros(max_it, 1);

    % ===== Main loop =====
    for i = 1:max_it

        % --- Column selection via LU pivoting ---
        t1 = tic;
        [J, start_idx_col] = idx_LU(ResL, block_size, J, start_idx_col);
        cidx = J(1:i * block_size);
        recent_cols = J((i-1) * block_size + 1:i * block_size);
        C_update = A(:, recent_cols);

        % Residual for row selection
        if i > 1
            if ns_var.svdfree
                Ap = C * (T \ (Q' * R(:, recent_cols))); % if using LU: replace Q'* with Q\
            else
                Ap = C * (T \ (Q  \ R(:, recent_cols))); % if using LU: replace Q\ with Q'*
            end
            ResR = C_update - Ap;
        else
            ResR = C_update;
        end

        % --- Row selection via LU pivoting ---
        [I, start_idx_row] = idx_LU(ResR', block_size, I, start_idx_row);
        C = A(:, cidx);
        ridx = I(I > 0);
        R_update = A(I((i-1) * block_size + 1:i * block_size), :);
        R = A(ridx, :);

        % --- CUR approximation error estimate ---
        [Q, T] = qr(full(C(ridx, :)));          % if using LU: [Q,T] = lu(full(C(ridx,:)))
        sketch = SA(:, cidx);
        Est = ((sketch / T) * Q') * R;        % if using LU: replace *Q' with /Q
        ResL = SA - Est;
        errors(i) = estimated_2norm(ResL);
        R_new = [R_new; R_update];
        timings.timing_approx = timings.timing_approx + toc(t1);

        % --- Decide whether to trigger preconditioning ---
        % Precondition if: error dropped significantly, below tolerance, or last iteration
        t2 = tic;
        prec_opt = err_dist > ns_var.tol_errdist * (errors(i) - err_tol) || ...
                   errors(i) < err_tol || i == max_it;
        if prec_opt
            err_dist = errors(i) - err_tol;
        end

        % --- QR factorization update ---
        if prec_opt
            if errors(i) > 1e-7 * anorm
                Rc = try_chol(@() chol(full(C'*C)), @() sketchol_qr(C, 0, 1));
            else
                [~, Rc] = sketchol_qr(C, 0, 1);
            end
        end
        if issparse(A)
            Rr = try_chol(@() chol(full(R*R')), @() sketchol_qr(R', 0, 1));
        else
            if i == 1, [Qr, Rr] = sketchol_qr(R');
            else,      [Qr, Rr] = block_qr_append(Qr, Rr, R_new'); end
        end
        R_new = [];
        timings.timing_aplicurprec = timings.timing_aplicurprec + toc(t2);

        if ~prec_opt, continue; end

        % --- Build preconditioner ---
        t3 = tic;
        Csub = C(ridx, :);
        if ns_var.svdfree
            % SVD-free: apply preconditioner implicitly via CUR factor structure
            targetlevel = errors(i);
            if issparse(R)
                pre     = @(x) pre_matvec_sparse(x, R, Csub, Rc, Rr, targetlevel);
                pret    = @(x) pret_matvec_sparse(x, R, Csub, Rc, Rr, targetlevel);
                preorig = @(x) preorig_matvec_sparse(x, R, Rr, Q, T, Rc, targetlevel);
            else
                Qrt = Qr';
                pre     = @(x) pre_matvec(x, Qr, Qrt, Csub, Rc, Rr, targetlevel);
                pret    = @(x) pret_matvec(x, Qr, Qrt, Csub, Rc, Rr, targetlevel);
                preorig = @(x) preorig_matvec(x, Qr, Qrt, Rr, Q, T, Rc, targetlevel);
            end
        else
            % SVD-based: compute SVD of M to get optimal scaling
            M = Rc * (T \ (Q \ Rr'));
            [~, Lhat, Vsmall] = svd(full(M), 0);
            Lhat = sqrt(Lhat*Lhat + mu^2 * eye(size(Lhat, 1)));
            minL = sqrt(Lhat(end,end)^2 + mu^2);
            if issparse(R)
                Vhat = @(x) R'*(Rr\(Vsmall*x));
                Vhatt = @(x) Vsmall'*(Rr'\(R*x));
            else
                Vhat = @(x) Qr*(Vsmall*x);
                Vhatt = @(x) Vsmall'*(Qr'*x);
            end
            pre     = @(x) Vhat((minL      ./diag(Lhat) - 1).*(Vhatt(x))) + x;
            preorig = @(x) Vhat((diag(Lhat)./minL       - 1).*(Vhatt(x))) + x;
            pret = pre;  % preconditioner is self-adjoint in SVD-based variant
        end
        matvec = @(x) apply(pre(x));
        timings.timing_aplicurprec = timings.timing_aplicurprec + toc(t3);

        if verbose, fprintf("i = %d\t errors(i) = %.2e\t", i, errors(i)); end

        % --- Initialize warm start ---
        if i == 1
            y0 = zeros(cols, 1);
            timestamps = toc(t0);
            timings.timing_aplicurlsqr = 0;
        else
            y0 = preorig(xs(:, idx));
            timestamps = [timestamps, toc(t0)];
            ts_resvecs(ts_idx + 1) = ts_resvecs(ts_idx);
            ts_xs(:, ts_idx + 1) = ts_xs(:, ts_idx);
            ts_idx = ts_idx + 1;
        end

        % --- Run LSQR ---
        t4 = tic;
        if errors(i) < err_tol || i == max_it
            % Final run: standard LSQR without early stopping
            fields_to_remove = {'minL', 'tol_cvgrate'};
            params_lsqr = rmfield(params_lsqr, fields_to_remove(isfield(params_lsqr, fields_to_remove)));
            params_lsqr.maxit = final_max_it;
            if ~ns_var.svdfree, fprintf("mu = %.2e\t", mu); end
        else
            % Intermediate run: early stopping via minL criterion
            params_lsqr.maxit = min(size(C, 2), final_max_it);
            if ns_var.svdfree
                params_lsqr.minL = errors(i);
            else
                params_lsqr.minL = minL;
            end
            fprintf("minL = %.2e\t", params_lsqr.minL);
        end
        [ys_lsqr, ~, resvec_lsqrp, ~, ts_lsqr, flag] = ...
            lsqr_scheduled(cols, apply, applyt, pre, pret, bmu, amunorm, params_lsqr, ...
            summary(matvec), y0, true);
        timings.timing_aplicurlsqr = timings.timing_aplicurlsqr + toc(t4);

        timestamps = [timestamps, timestamps(end) + ts_lsqr];
        xs_lsqr_new = pre(ys_lsqr);
        n_new = length(resvec_lsqrp);

        % --- Append results ---
        if i == 1
            resvecs(1:n_new) = resvec_lsqrp;
            ts_resvecs(1) = resvec_lsqrp(1);
            ts_resvecs(1+(1:n_new)) = resvec_lsqrp;
            xs(:, 1:n_new) = xs_lsqr_new;
            ts_xs(:, 1) = xs_lsqr_new(:, 1);
            ts_xs(:, 1+(1:n_new)) = xs_lsqr_new;
            idx = n_new; ts_idx = 1 + n_new;
        else
            resvecs(idx+(1:n_new)) = resvec_lsqrp;
            ts_resvecs(ts_idx+(1:n_new)) = resvec_lsqrp;
            xs(:, idx+(1:n_new)) = xs_lsqr_new;
            ts_xs(:, ts_idx+(1:n_new)) = xs_lsqr_new;
            idx = idx + n_new; ts_idx = ts_idx + n_new;
        end

        if errors(i) < err_tol || flag == 1 || flag == 3, break; end
    end

    % ===== Trim preallocated arrays =====
    ls = size(C, 2);
    resvecs = resvecs(1:idx);
    ts_resvecs = ts_resvecs(1:ts_idx);
    xs = xs(:, 1:idx);
    ts_xs = ts_xs(:, 1:ts_idx);

    timings.timing_aplicurprec = timings.timing_aplicurprec + timings.timing_approx;
end



% =========================================================================
% SVD-free preconditioner matvecs
% The preconditioner is applied implicitly using the CUR factor structure,
% avoiding the SVD of M = Rc*(T\(Q\Rr')) used in aplicur/aplicur_singleshot.
% =========================================================================

function y = pre_matvec(x, Qr, Qrt, Csub, Rc, Rr, level)
    % Forward preconditioner application (dense)
    Qrtx = Qrt * x;
    u    = Rr' \ (Csub * (Rc \ Qrtx));
    y    = x + Qr * (level * u - Qrtx);
end

function y = pre_matvec_sparse(x, R, Csub, Rc, Rr, level)
    % Forward preconditioner application (sparse)
    Qrtx = Rr' \ (R * x);
    u    = Rr' \ (Csub * (Rc \ Qrtx));
    y    = x + R' * (Rr \ (level * u - Qrtx));
end

function y = pret_matvec(x, Qr, Qrt, Csub, Rc, Rr, level)
    % Adjoint preconditioner application (dense)
    Qrtx = Qrt * x;
    u    = Rc' \ (Csub' * (Rr \ Qrtx));
    y    = x + Qr * (level * u - Qrtx);
end

function y = pret_matvec_sparse(x, R, Csub, Rc, Rr, level)
    % Adjoint preconditioner application (sparse)
    Qrtx = Rr' \ (R * x);
    u    = Rc' \ (Csub' * (Rr \ Qrtx));
    y    = x + R' * (Rr \ (level * u - Qrtx));
end

function y = preorig_matvec(x, Qr, Qrt, Rr, Q, T, Rc, level)
    % Inverse preconditioner application (dense) — used for warm-start
    a = Qrt * x;
    u = Rc * (T \ (Q' * (Rr' * a)));
    y = x + Qr * (u / level - a);
end

function y = preorig_matvec_sparse(x, R, Rr, Q, T, Rc, level)
    % Inverse preconditioner application (sparse) — used for warm-start
    Rx = R * x;
    a  = Rr' \ Rx;
    u  = Rc * (T \ (Q' * Rx));
    y  = x + R' * (Rr \ (u / level - a));
end


% =========================================================================
% Helper functions
% =========================================================================

function [I, start] = idx_LU(Res, block_size, I, start)
    % Select block_size row indices from Res via LU pivoting
    Res(:, I(I > 0)) = 0;
    [~, ~, P] = lu(full(Res'), "vector");
    end_idx = start + block_size - 1;
    I(start:end_idx) = P(1:block_size)';
    start = end_idx + 1;
end

function [Q, R] = block_qr_append(Q, R, C_new)
    % Incrementally update QR factorization with new columns C_new
    Proj1 = Q' * C_new;
    Res   = C_new - Q * Proj1;
    Proj2 = Q' * Res;
    Res   = Res - Q * Proj2;
    [Q_Res, R_Res] = sketchol_qr(Res);
    Q = [Q, Q_Res];
    R = [R, Proj1+Proj2; zeros(size(R_Res,1), size(R,2)), R_Res];
end

function R = try_chol(chol_func, fallback_func)
    % Attempt Cholesky; fall back to sketch-based QR if it fails
    try
        R = chol_func();
    catch
        [~, R] = fallback_func();
    end
end

function SA = SRFT(A, s)
    % Subsampled Randomized Fourier Transform sketch
    m = size(A, 1);
    d = sign(randn(m, 1));
    if isreal(A)
        SA = dct(d .* A); SA(1,:) = SA(1,:) / sqrt(2);
    else
        SA = fft(d .* A);
    end
    SA = SA(randsample(m, s), :) * sqrt(m/s);
end

function errorest = estimated_2norm(A)
    % Estimate 2-norm of A via 10 random power iterations
    errormax = 0;
    for i = 1:10
        v = randn(size(A,2), 1); v = v / norm(v);
        errormax = max(errormax, norm(A*v));
    end
    errorest = 10 * sqrt(2/pi) * errormax;
end