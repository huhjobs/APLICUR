function run_npcg(varargin)
    close all;
    warning('off', 'all');
    set(0, 'DefaultFigureWindowStyle', 'docked');
    addpath('../baselines/');

    % ===== Input parsing =====
    p = inputParser;
    p.addParameter('SEED',           randi([0, 100]), @(x) isnumeric(x) && isscalar(x));
    p.addParameter('DISPLAYTIMINGS', true,            @(x) islogical(x));
    p.addParameter('ALGNAME',        'npcg',          @(x) ischar(x));
    p.addParameter('L0',             5,               @(x) isnumeric(x) && isscalar(x));
    p.addParameter('LCAP',           Inf,             @(x) isnumeric(x) && isscalar(x));
    p.addParameter('ERRTOL',         1e-16,           @(x) isnumeric(x) && isscalar(x));
    p.addParameter('dataA',          [],              @(x) isnumeric(x) && ismatrix(x));
    p.addParameter('dataSA',         [],              @(x) isnumeric(x) && isvector(x));
    p.addParameter('datab',          [],              @(x) isnumeric(x) && isvector(x));
    p.addParameter('MU',             1e-3,            @(x) isnumeric(x) && isscalar(x));
    p.addParameter('tol',            1e-10,           @(x) isnumeric(x) && isscalar(x));
    p.addParameter('maxit',          500,             @(x) isnumeric(x) && isscalar(x));
    p.addParameter('OUTDIR',         './results/',    @(x) ischar(x));
    p.parse(varargin{:});

    % ===== Unpack parameters =====
    seed    = p.Results.SEED;   rng(seed);
    algname = p.Results.ALGNAME;
    A       = p.Results.dataA;
    b       = p.Results.datab;
    SA      = p.Results.dataSA;
    [m, n]  = size(A);
    mu      = p.Results.MU;
    l0      = p.Results.L0;
    lcap    = min([m, n, p.Results.LCAP]);
    errtol  = p.Results.ERRTOL;
    outdir  = p.Results.OUTDIR;

    DEBUG = false;

    % ===== Setup output =====
    if ~exist(outdir, 'dir'), mkdir(outdir); end
    diary(sprintf('%sterminal_output_%s.txt', outdir, algname));

    parsedData = rmfield(p.Results, {'dataA', 'datab'});

    % ===== Problem setup =====
    % For regularized case, augment the system; otherwise use A, b directly
    if mu
        Amu = [A; mu * eye(n)];
        bmu = [b; zeros(n, 1)];
    else
        Amu = A;
        bmu = b;
    end

    params_cg.tol   = p.Results.tol;
    params_cg.maxit = min(n, 5000);

    apply  = @(x) Amu * x;
    applyt = @(x) Amu' * x;

    fprintf("\n================Nystrom PCG Run================\n");
    fprintf("errtol = %.2e, l0 = %d\n", errtol, l0);

    % ===== Run Nystrom PCG =====
    % Guard: skip if smallest singular value is negligible
    if sqrt(SA(end)^2 + mu^2) < 1e-8
        fprintf("Skipping: smallest singular value too small (sqrt(SA(end)^2+mu^2) < 1e-8).\n");
        diary off; return;
    end

    try
        timestart = tic;
    
        % --- Nyström approximation of A'A ---
        gramvec = @(x) A' * (A * x);
        sketch  = 'sparse';
        if algname == "npcg_gaussian", sketch = 'gaussian'; end
    
        [Uhat, Lhat, timing_nysprec] = npcg( ...
            gramvec, n, l0, lcap, 5, errtol^2, sketch);
    
        % --- Build preconditioner ---
        minL = Lhat(end, end);
        if mu
            Lhat = Lhat + mu^2 * eye(size(Lhat, 1));
            minL = minL + mu^2;
        end
        pre = @(x) Uhat * ((minL ./ diag(Lhat) - 1) .* (Uhat' * x)) + x;
    
        % --- Run PCG on normal equations A'Ax = A'b ---
        apply_gram = @(x) applyt(apply(x));
        summary_cg = @(x) norm(apply(x) - bmu);
        [xs_np, ~, resvec_np, timing_npcg, ts_npcg, flag_np] = ...
            cg_ethan(apply_gram, pre, Amu' * bmu, params_cg, summary_cg);
    
        fprintf("Total time:                   %.3f sec\n", toc(timestart));
        fprintf("Nyström preconditioning time: %.3f sec\n", timing_nysprec.total);
        fprintf("PCG time:                     %.3f sec\n\n", timing_npcg.total);
    
        % --- Collect output ---
        output.timings.solve  = timing_npcg;
        output.timings.prec   = timing_nysprec;
        output.total_timings  = timing_npcg.total + timing_nysprec.total;
        output.resvecs        = resvec_np;
        output.timestamps     = ts_npcg;
        output.flag           = flag_np;
        output.ls             = size(Lhat, 1);
    
        % Residuals without regularization
        output.resvecs_woreg = vecnorm(A * xs_np - b, 2, 1)';
    
        % Singular values of preconditioned operator (requires DEBUG = true)
        if DEBUG
            output.precsv   = svd(apply_gram(pre(eye(n))));
            output.preccond = output.precsv(1) / output.precsv(end);
            fprintf("Cond(A/P_NPCG) = %.2e\n", output.preccond);
        end
    catch ME
        fprintf("NPCG failed: %s\n\n", ME.message);
        output.ls = NaN;
    end

    % ===== Save timings and output =====
    if p.Results.DISPLAYTIMINGS
        fid = fopen(fullfile(outdir, sprintf('timing_detail_%s.txt', algname)), 'w');
        printStruct(output.timings, '', '', fid);
        fclose(fid);
    end

    output.parsed = parsedData;
    save(sprintf('./%s/output_%s_%d.mat', outdir, algname, seed), 'output', '-v7.3');

    diary off;
end