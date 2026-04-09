function run_bdp(varargin)
    close all;
    warning('off', 'all');
    set(0, 'DefaultFigureWindowStyle', 'docked');
    addpath('./baselines/');

    % ===== Input parsing =====
    p = inputParser;
    p.addParameter('SEED',           randi([0, 100]), @(x) isnumeric(x) && isscalar(x));
    p.addParameter('DISPLAYTIMINGS', true,            @(x) islogical(x));
    p.addParameter('ALGNAME',        'bdp',           @(x) ischar(x));
    p.addParameter('BLENGAMMA',      2,               @(x) isnumeric(x) && isscalar(x));
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
    outdir  = p.Results.OUTDIR;

    DEBUG = false;

    % ===== Setup output =====
    if ~exist(outdir, 'dir'), mkdir(outdir); end
    diary(sprintf('%sterminal_output_%s.txt', outdir, algname));

    parsedData = rmfield(p.Results, {'dataA', 'datab'});

    % ===== Problem setup =====
    Amu     = [A; mu * eye(n)];
    bmu     = [b; zeros(n, 1)];
    amunorm = sqrt(SA(1)^2 + mu^2);

    % Cap gamma so sketch size doesn't exceed m+n
    gamma = min(p.Results.BLENGAMMA, (m + n) / n);

    params_lsqr.tol   = p.Results.tol;
    params_lsqr.maxit = min(n, 5000);

    apply   = @(x) Amu * x;
    applyt  = @(x) Amu' * x;
    summary = @(mv) @(x) norm(mv(x) - bmu);

    fprintf("\n================Blendenpik Run================\n");
    fprintf("Algorithm: %s\n", algname);

    % ===== Build Blendenpik preconditioner =====
    % Each variant selects a sketch type and optionally computes an initial
    % solution x0 from the sketched system (isp = improve start point).
    timestart = tic;
    y0 = zeros(n, 1);
    switch algname
        case 'bdp'
            [P_BDP, timing_bdpprec] = blendenpik_matlab(Amu, 'SRFT', gamma);

        case 'bdp_isp'
            [P_BDP, timing_bdpprec, x0] = blendenpik_matlab(Amu, 'SRFT', gamma, true, bmu);
            y0 = P_BDP * x0;

        case 'bdp_sparse'
            [P_BDP, timing_bdpprec] = blendenpik_matlab(Amu, 'SparseEmbedding', gamma);

        case 'bdp_sparse_isp'
            [P_BDP, timing_bdpprec, x0] = blendenpik_matlab(Amu, 'SparseEmbedding', gamma, true, bmu);
            y0 = P_BDP * x0;

        otherwise
            error('Unknown algorithm: %s', algname);
    end
    timing_bdpprec.total = toc(timestart);

    % ===== Run preconditioned LSQR =====
    pre    = @(x) P_BDP \ x;
    pret   = @(x) P_BDP' \ x;
    matvec = @(x) apply(pre(x));

    [ys_bdp, ~, resvec_bdp, timing_bdplsqr, ts_bdp, flag_bdp] = ...
        lsqr_ethan_new(n, apply, applyt, pre, pret, bmu, amunorm, ...
        params_lsqr, summary(matvec), y0, false);

    fprintf("Total time:               %.3f sec\n", toc(timestart));
    fprintf("Preconditioning time:     %.3f sec\n", timing_bdpprec.total);
    fprintf("LSQR time:                %.3f sec\n", timing_bdplsqr.total);

    % ===== Collect output =====
    output.timings.solve  = timing_bdplsqr;
    output.timings.prec   = timing_bdpprec;
    output.total_timings  = timing_bdplsqr.total + timing_bdpprec.total;
    output.resvecs        = resvec_bdp;
    output.timestamps     = ts_bdp;
    output.flag           = flag_bdp;

    % Residuals without regularization
    output.resvecs_woreg = vecnorm(A * pre(ys_bdp) - b, 2, 1)';
    

    % Singular values of preconditioned operator (requires DEBUG = true)
    if DEBUG
        output.precsv  = svd(matvec(eye(n)));
        output.preccond = output.precsv(1) / output.precsv(end);
        fprintf("Cond(A/P_BDP) = %.2e\n", output.preccond);
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