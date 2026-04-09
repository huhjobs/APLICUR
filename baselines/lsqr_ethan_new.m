function [ys,iter,stats,timing,timestamps,flag] = lsqr_ethan_new(n,apply,applyt,pre,pret,b,anorm,params,varargin)
% LSQR with preconditioning and adaptive stopping criteria.
% Based on: https://github.com/eepperly/Stable-Randomized-Least-Squares/blob/main/code/mylsqr.m
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
%   varargin{1} - (optional) summary function @(x) -> stats row vector
%   varargin{2} - (optional) initial guess x0

    d = 5;  % window size for projected residual stopping criterion
    timing.total = 0;
    
    tol   = params.tol;
    maxit = params.maxit;
    preconditioned = ~(~anorm);

    matvec = @(x) apply(pre(x)); adjvec = @(x) pret(applyt(x));

    summary = [];
    if ~isempty(varargin)
        summary = varargin{1}; % Statistic function handle
    end
    
    % Process initial guess for x if provided
    if length(varargin) > 1 && ~isempty(varargin{2}) && norm(varargin{2}) ~= 0
        y = varargin{2};
    else
        y = zeros(n,1);
    end

    lsqrtic = tic;
    ys = y; timestamps = toc(lsqrtic);
    stats = []; phisqs = []; phisqsums = []; tolerances = [];
    
    bnorm = norm(b);
    u = b - matvec(y); beta = norm(u); u = u / beta; % Normalize initial residual
    
    flag = 0;
    
    v = adjvec(u); alpha = norm(v); v = v / alpha; % The first search direction
    w = v; % Store previous search direction
    phibar = beta; rhobar = alpha; % Initialize projection scalars
    initgradnorm = alpha * beta;

    % Begin LSQR iteration
    for iter = 1:maxit
        % Lanczos Bidiagonalization
        u = matvec(v) - alpha*u; beta = norm(u); u = u / beta; % beta_{iter+1}
        if ~preconditioned, if iter == 1, anorm = sqrt(alpha^2 + beta^2); end; end
        v = adjvec(u) - beta*v; alpha = norm(v); v = v / alpha; % alpha_{iter+1}
        if ~preconditioned, anorm = norm([anorm alpha beta],'fro'); end

        % Given's Rotation
        rho = sqrt(rhobar^2 + beta^2); % Compute new rho
        c = rhobar / rho; s = beta / rho;
        theta = s * alpha;  rhobar = - c * alpha;
        phi   = c * phibar; phibar = s * phibar;
        
        % Update solution
        y = y + (phi/rho) * w;
        w = v - (theta/rho) * w;

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
        % Convergence Check
        if ~isempty(summary), ys(:,end+1) = y; end %#ok<AGROW> 
        tolerances(end+1) = tol * (anorm * norm(pre(y)) + bnorm);
        if phibar * alpha * abs(c) < tol * initgradnorm % lsqr_ethan code gradnormest/(IRinitgradnormest)
            flag = 1; break;
        end
        if phibar < tol * bnorm % residualest/initialresidual
            flag = 2; break;
        end
        if iter > d
            if phisqsums(end) <= tol * phisqsums(1) % stopcrit paper LStest1 gradnormest/apnormest
                flag = 3; break;
            end
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
    end
end