function [ys,iter,stats,timing,timestamps,flag] = lsqr_scheduled(n,apply,applyt,pre,pret,b,anorm,params,varargin)
% LSQR with preconditioning and adaptive stopping criteria.
% Extends lsqr_ethan_new with two additional early stopping criteria
% (flags 5 & 6) activated by setting params.tol_cvgrate and params.minL.
%
% Inputs:
%   n           - problem dimension
%   apply       - function handle for A*x
%   applyt      - function handle for A'*x
%   pre         - function handle for preconditioner P*x
%   pret        - function handle for adjoint preconditioner P'*x
%   b           - right-hand side vector
%   anorm       - estimated norm of A (set to 0 if unknown; will be estimated)
%   params      - struct with fields: tol, maxit
%                 optional fields: tol_cvgrate, minL (activate flags 5 & 6)
%   varargin{1} - (optional) summary function @(x) -> stats row vector
%   varargin{2} - (optional) initial guess x0

    d = 5;  % window size for projected residual stopping criterion

    timing.total = 0;
    tol   = params.tol;
    maxit = params.maxit;
    preconditioned = ~(~anorm);

    matvec = @(x) apply(pre(x)); adjvec = @(x) pret(applyt(x));

    % ===== Optional arguments =====
    summary = [];
    if ~isempty(varargin), summary = varargin{1}; end

    if length(varargin) > 1 && ~isempty(varargin{2}) && norm(varargin{2}) ~= 0
        y = varargin{2};
    else
        y = zeros(n,1);
    end
    
    % ===== Initialization =====
    lsqrtic = tic;
    ys = y; timestamps = toc(lsqrtic);
    stats = []; phisqs = []; phisqsums = []; tolerances = [];

    bnorm = norm(b);
    u = b - matvec(y); beta = norm(u); u = u / beta;

    flag = 0; phibars = [];
    v = adjvec(u); alpha = norm(v); v = v / alpha;
    w = v;
    phibar = beta; rhobar = alpha;
    initgradnorm = alpha * beta;

    % ===== LSQR iterations =====
    for iter = 1:maxit
        
        % --- Lanczos bidiagonalization ---
        u = matvec(v) - alpha*u; beta = norm(u); u = u / beta;
        if ~preconditioned, if iter == 1, anorm = sqrt(alpha^2 + beta^2); end; end
        v = adjvec(u) - beta*v; alpha = norm(v); v = v / alpha;
        if ~preconditioned, anorm = norm([anorm alpha beta],'fro'); end

        % --- Givens rotation ---
        rho = sqrt(rhobar^2 + beta^2);
        c = rhobar / rho; s = beta / rho;
        theta = s * alpha;  rhobar = -c * alpha;
        phi   = c * phibar; phibar = s * phibar;

        % --- Solution update ---
        y = y + (phi/rho) * w;
        w = v - (theta/rho) * w;

        % --- Projected residual norm estimate (windowed) ---
        phisqs(end+1) = phi^2;
        if iter > d
            if iter == d + 1
                sq_sum = sum(phisqs(iter-d+1:iter));
            else
                sq_sum = sq_sum - phisqs(iter-d) + phisqs(iter);
                if sq_sum < 0, sq_sum = sum(phisqs(iter-d+1:iter)); end
            end
            phisqsums(end+1) = sqrt(sq_sum);
        end

        timestamps(end+1) = toc(lsqrtic);
        ys(:,end+1)       = y; %#ok<AGROW>
        tolerances(end+1) = tol * (anorm * norm(pre(y)) + bnorm);

        % --- Early stopping via convergence rate (flags 5 & 6) ---
        % Activated only when params.tol_cvgrate and params.minL are set.
        % Flag 5: convergence rate has slowed below initial rate.
        % Flag 6: absolute decrease in phibar is smaller than params.minL.
        phibars(end+1) = phibar;
        if isfield(params, 'tol_cvgrate')
            if iter == 1
                init_cvg_rate = log10(bnorm) - log10(phibar);
            else
                cvg_rate = log10(phibars(end-1)) - log10(phibars(end));
                if params.tol_cvgrate * cvg_rate < init_cvg_rate
                    flag = 5; break;
                end
                cvg_diff = phibars(end-1) - phibars(end);
                if cvg_diff < params.minL
                    flag = 6; break;
                end
            end
        end

        % --- Standard stopping criteria ---
        if phibar * alpha * abs(c) < tol * initgradnorm, flag = 1; break; end % gradient norm estimate
        if phibar < tol * bnorm, flag = 2; break; end % residual estimate
        if iter > d
            if phisqsums(end) <= tol * phisqsums(1), flag = 3; break; end % projected residual decrease
        end
        if iter == maxit, flag = 4; break; end
    end

    timing.total = toc(lsqrtic);

    if ~isempty(summary)
        for i = 1:iter+1
            stats(end+1,:) = summary(ys(:,i));
        end
    end

    switch flag
        case 1, fprintf("Terminated with acceptable LS solution in %d iterations (ethan).\n", iter);
        case 2, fprintf("Terminated with acceptable residual in %d iterations.\n", iter);
        case 3, fprintf("Terminated with acceptable LS solution in %d iterations (projected).\n", iter);
        case 4, fprintf("Failed to achieve desired back stability in %d iterations.\n", maxit);
        case 5, fprintf("Terminated with acceptable LS solution in %d iterations (ilsqr_cvgrate).\n", iter);
        case 6, fprintf("Terminated with acceptable LS solution in %d iterations (ilsqr_minL).\n", iter);
    end
end