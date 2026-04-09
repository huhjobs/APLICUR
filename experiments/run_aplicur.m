function run_aplicur(varargin)
    close all;
    warning('off', 'all');
    set(0, 'DefaultFigureWindowStyle', 'docked');
    addpath('../algorithms/');

    % ===== Input parsing =====
    p = inputParser;
    p.addParameter('SEED',           randi([0, 100]), @(x) isnumeric(x) && isscalar(x));
    p.addParameter('DISPLAYTIMINGS', true,            @(x) islogical(x));
    p.addParameter('ALGNAME',        'aplicur',       @(x) ischar(x));
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
    [m, n]  = size(A);
    mu      = p.Results.MU;
    errtol  = p.Results.ERRTOL;
    l0      = p.Results.L0;
    lcap    = min([m, n, p.Results.LCAP]);
    outdir  = p.Results.OUTDIR;

    DEBUG = false;

    % ===== Setup output =====
    if ~exist(outdir, 'dir'), mkdir(outdir); end
    diary(sprintf('%sterminal_output_%s.txt', outdir, algname));

    parsedData = rmfield(p.Results, {'dataA', 'datab'});

    % ===== Problem setup =====
    Amu = [A; mu * eye(n)];
    bmu = [b; zeros(n, 1)];

    params_lsqr.tol   = p.Results.tol;
    params_lsqr.maxit = min(n, 5000);

    block_size = l0;
    max_it     = min(ceil(lcap / block_size), floor(n / block_size));
    ns_var.sketch = 'SparseEmbedding';

    fprintf("\n================APLICUR Run================\n");
    fprintf("Algorithm: %s\n", algname);
    fprintf("block_size=%d, errtol=%.2e\n", block_size, errtol);

    % ===== Dispatch to algorithm variant =====
    % Each variant sets max_it/errtol overrides if needed, then calls the
    % appropriate underlying function (aplicur or aplicur_singleshot/svdfree).
    timestart = tic;
    switch algname
        case 'aplicur'
            [xs_aplicur, ts_xs_aplicur, resvecs_aplicur, ts_resvecs_aplicur, ...
                timing_aplicur, ts_aplicur, ls_aplicur, flag_aplicur, matvec] = ...
                aplicur(A, mu, Amu, bmu, block_size, errtol, max_it, params_lsqr, true, ns_var);

        case 'aplicur_svdfree'
            ns_var.svdfree = true;
            [xs_aplicur, ts_xs_aplicur, resvecs_aplicur, ts_resvecs_aplicur, ...
                timing_aplicur, ts_aplicur, ls_aplicur, flag_aplicur, matvec] = ...
                aplicur(A, mu, Amu, bmu, block_size, errtol, max_it, params_lsqr, true, ns_var);

        case 'aplicur_singleshot'
            ns_var.tol_errdist = Inf;   % disable adaptive scheduling
            [xs_aplicur, ts_xs_aplicur, resvecs_aplicur, ts_resvecs_aplicur, ...
                timing_aplicur, ts_aplicur, ls_aplicur, flag_aplicur, matvec] = ...
                aplicur_singleshot(A, mu, Amu, bmu, block_size, errtol, max_it, params_lsqr, true, ns_var);

        case 'aplicur_fixed'        % fixed rank = 20% of n
            max_it     = 1;
            block_size = round(0.2 * n);
            [xs_aplicur, ts_xs_aplicur, resvecs_aplicur, ts_resvecs_aplicur, ...
                timing_aplicur, ts_aplicur, ls_aplicur, flag_aplicur, matvec] = ...
                aplicur_singleshot(A, mu, Amu, bmu, block_size, errtol, max_it, params_lsqr, true, ns_var);

        case 'aplicur_fixed_under'  % fixed rank = 10% of n
            max_it     = 1;
            block_size = round(0.1 * n);
            [xs_aplicur, ts_xs_aplicur, resvecs_aplicur, ts_resvecs_aplicur, ...
                timing_aplicur, ts_aplicur, ls_aplicur, flag_aplicur, matvec] = ...
                aplicur_singleshot(A, mu, Amu, bmu, block_size, errtol, max_it, params_lsqr, true, ns_var);

        case 'aplicur_fixed_over'   % fixed rank = 30% of n
            max_it     = 1;
            block_size = round(0.3 * n);
            [xs_aplicur, ts_xs_aplicur, resvecs_aplicur, ts_resvecs_aplicur, ...
                timing_aplicur, ts_aplicur, ls_aplicur, flag_aplicur, matvec] = ...
                aplicur_singleshot(A, mu, Amu, bmu, block_size, errtol, max_it, params_lsqr, true, ns_var);

        case 'aplicur_correct'      % adaptive, runs until errtol=0 (up to 10 rounds)
            max_it = 10; errtol = 0;
            [xs_aplicur, ts_xs_aplicur, resvecs_aplicur, ts_resvecs_aplicur, ...
                timing_aplicur, ts_aplicur, ls_aplicur, flag_aplicur, matvec] = ...
                aplicur(A, mu, Amu, bmu, block_size, errtol, max_it, params_lsqr, true, ns_var);

        case 'aplicur_under'        % adaptive, fewer rounds (5)
            max_it = 5; errtol = 0;
            [xs_aplicur, ts_xs_aplicur, resvecs_aplicur, ts_resvecs_aplicur, ...
                timing_aplicur, ts_aplicur, ls_aplicur, flag_aplicur, matvec] = ...
                aplicur(A, mu, Amu, bmu, block_size, errtol, max_it, params_lsqr, true, ns_var);

        case 'aplicur_over'         % adaptive, more rounds (15)
            max_it = 15; errtol = 0;
            [xs_aplicur, ts_xs_aplicur, resvecs_aplicur, ts_resvecs_aplicur, ...
                timing_aplicur, ts_aplicur, ls_aplicur, flag_aplicur, matvec] = ...
                aplicur(A, mu, Amu, bmu, block_size, errtol, max_it, params_lsqr, true, ns_var);

        otherwise
            error('Unknown algorithm: %s', algname);
    end

    fprintf("Total time:            %.3f sec\n", toc(timestart));
    fprintf("Preconditioning time:  %.3f sec\n", timing_aplicur.timing_aplicurprec);
    fprintf("LSQR time:             %.3f sec\n", timing_aplicur.timing_aplicurlsqr);

    % ===== Collect output =====
    output.timings.total            = timing_aplicur.timing_aplicurlsqr;
    output.timings.prec.total       = timing_aplicur.timing_aplicurprec;
    output.timings.prec.sketch_time = timing_aplicur.sketch;
    output.total_timings            = timing_aplicur.timing_aplicurlsqr + timing_aplicur.timing_aplicurprec;
    output.blocksize   = block_size;
    output.ls          = ls_aplicur;
    output.resvecs     = resvecs_aplicur;
    output.tsresvecs   = ts_resvecs_aplicur;
    output.timestamps  = ts_aplicur;
    output.flag        = flag_aplicur;

    % Residuals without regularization
    output.resvecs_woreg    = vecnorm(A * xs_aplicur    - b, 2, 1)';
    output.resvecs_woreg_ts = vecnorm(A * ts_xs_aplicur - b, 2, 1)';
    

    % Singular values of preconditioned operator (requires DEBUG = true)
    if DEBUG, output.precsv = svd(matvec(eye(n))); end

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