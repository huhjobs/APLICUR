function run_lsqr(varargin)
    close all;
    warning('off', 'all');
    set(0, 'DefaultFigureWindowStyle', 'docked');
    addpath('../baselines/');

    % ===== Input parsing =====
    p = inputParser;
    p.addParameter('SEED',           randi([0, 100]), @(x) isnumeric(x) && isscalar(x));
    p.addParameter('DISPLAYTIMINGS', true,            @(x) islogical(x));
    p.addParameter('dataA',          [],              @(x) isnumeric(x) && ismatrix(x));
    p.addParameter('dataSA',         [],              @(x) isnumeric(x) && isvector(x));
    p.addParameter('datab',          [],              @(x) isnumeric(x) && isvector(x));
    p.addParameter('MU',             1e-3,            @(x) isnumeric(x) && isscalar(x));
    p.addParameter('tol',            1e-10,           @(x) isnumeric(x) && isscalar(x));
    p.addParameter('maxit',          5000,            @(x) isnumeric(x) && isscalar(x));
    p.addParameter('OUTDIR',         './results/',    @(x) ischar(x));
    p.parse(varargin{:});

    % ===== Unpack parameters =====
    seed   = p.Results.SEED;   rng(seed);
    A      = p.Results.dataA;
    b      = p.Results.datab;
    [~, n] = size(A);
    mu     = p.Results.MU;
    outdir = p.Results.OUTDIR;

    % ===== Setup output =====
    if ~exist(outdir, 'dir'), mkdir(outdir); end
    diary(sprintf('%sterminal_output_lsqr.txt', outdir));

    parsedData = rmfield(p.Results, {'dataA', 'datab'});

    % ===== Problem setup =====
    Amu = [A; mu * eye(n)];
    bmu = [b; zeros(n, 1)];

    params_lsqr.tol   = p.Results.tol;
    params_lsqr.maxit = min(n, p.Results.maxit);

    apply   = @(x) Amu * x;
    applyt  = @(x) Amu' * x;
    pre     = @(x) x;           % identity preconditioner (unpreconditioned LSQR)
    pret    = @(x) x;
    matvec  = @(x) apply(x);
    summary = @(mv) @(x) norm(mv(x) - bmu);

    fprintf("\n================LSQR Run================\n");

    % ===== Run unpreconditioned LSQR =====
    timestart = tic;
    [xs, ~, resvec, timing_lsqr, ts_lsqr, flag] = lsqr_ethan_new( ...
        n, apply, applyt, pre, pret, bmu, 0, params_lsqr, ...
        summary(matvec), zeros(n, 1), false);

    fprintf("Total time: %.3f sec\n",  toc(timestart));
    fprintf("LSQR time:  %.3f sec\n\n", timing_lsqr.total);

    % ===== Collect output =====
    output.timings       = timing_lsqr;
    output.total_timings = timing_lsqr.total;
    output.resvecs       = resvec;
    output.timestamps    = ts_lsqr;
    output.flag          = flag;

    % Residuals without regularization
    output.resvecs_woreg = vecnorm(A * xs - b, 2, 1)';
    

    % ===== Save timings and output =====
    if p.Results.DISPLAYTIMINGS
        fid = fopen(fullfile(outdir, 'timing_detail_lsqr.txt'), 'w');
        printStruct(output.timings, '', '', fid);
        fclose(fid);
    end

    output.parsed = parsedData;
    save(sprintf('./%s/output_lsqr_%d.mat', outdir, seed), 'output', '-v7.3');

    diary off;
end