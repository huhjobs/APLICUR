function [xs, iter, stats, timing, timestamps, flag] = cg_ethan(matvec, prec, b, params, varargin)
% Preconditioned Conjugate Gradient (PCG).
%
% Inputs:
%   matvec      - function handle for matrix-vector product A*x
%   prec        - function handle for preconditioner application M\x
%   b           - right-hand side vector
%   params      - struct with fields: tol, maxit
%   varargin{1} - (optional) summary function @(x) -> stats row vector
%   varargin{2} - (optional) initial iterate x0
%   varargin{3} - (optional) verbose flag (default: false)

    tol   = params.tol;
    maxit = params.maxit;

    % ===== Optional arguments =====
    summary = [];
    if ~isempty(varargin), summary = varargin{1}; end

    if length(varargin) > 1 && ~isempty(varargin{2}) && norm(varargin{2}) ~= 0
        x = varargin{2};
        r = b - matvec(x);
    else
        x = zeros(size(b));
        r = b;
    end

    verbose = false;
    if length(varargin) > 2 && ~isempty(varargin{3}), verbose = varargin{3}; end

    % ===== Initialization =====
    cgtic = tic;
    stats = []; xs = []; timestamps = [];
    if ~isempty(summary)
        xs(:, end+1)      = x;
        timestamps(end+1) = toc(cgtic);
    end

    bnorm = norm(b);
    z     = prec(r);
    p     = z;
    flag  = 4;

    % ===== PCG iterations =====
    for iter = 1:maxit
        if verbose, fprintf('%d\t%e\n', iter, norm(r) / bnorm); end

        v   = matvec(p);
        zr  = z' * r;
        eta = zr / (v' * p);
        x   = x + eta * p;
        r   = r - eta * v;
        z   = prec(r);
        p   = z + (z' * r / zr) * p;

        if ~isempty(summary)
            xs(:, end+1)      = x;          %#ok<AGROW>
            timestamps(end+1) = toc(cgtic); %#ok<AGROW>
        end

        if norm(r) <= tol * bnorm, flag = 1; break; end
    end

    timing.total = toc(cgtic);
    fprintf("Terminated in %d iterations (flag=%d).\n", iter, flag);

    if ~isempty(summary)
        for i = 1:iter+1
            stats(end+1, :) = summary(xs(:, i)); %#ok<AGROW>
        end
    end
end