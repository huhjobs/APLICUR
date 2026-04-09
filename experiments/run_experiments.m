function run_experiments(varargin)
    addpath('../utils/');

    % ===== Input parsing =====
    p = inputParser;
    p.addParameter('DATASETPATH', '../datasets/matfilename', @(x) ischar(x));
    p.addParameter('SUBDIR',      'tmp', @(x) ischar(x));
    p.addParameter('MU',          1e-4, @(x) isnumeric(x) && isscalar(x));
    p.addParameter('ERRTOL',      30*1e-4, @(x) isnumeric(x) && isscalar(x));
    p.parse(varargin{:});

    datasetpath = p.Results.DATASETPATH;
    subdir      = p.Results.SUBDIR;
    mu          = p.Results.MU;
    errtol      = p.Results.ERRTOL;

    % ===== Load dataset =====
    fprintf("Loading dataset from %s ...\n", datasetpath);

    datasetname = strsplit(datasetpath, '/');
    datasetname = strsplit(datasetname{end}, '.');
    datasetname = datasetname{1};

    dataset = load(datasetpath);
    dataA   = dataset.dataset.A;
    [m, n]  = size(dataA);
    fprintf("Dataset [%s] loaded: %d x %d, sparsity=%.3f\n", datasetname, m, n, nnz(dataA)/(m*n));

    % Singular values (precomputed or computed on the fly)
    if isfield(dataset.dataset, 'SA')
        dataSA = dataset.dataset.SA;
    else
        fprintf("Computing SVD ...\n");
        dataSA = svd(full(dataA));
    end

    % Right-hand side
    datab = dataset.dataset.b;

    % Reference solution (precomputed or solved directly)
    if isfield(dataset.dataset, 'x')
        datax = dataset.dataset.x;
    else
        fprintf("Computing reference solution via lsqminnorm ...\n");
        datax = lsqminnorm(dataA, datab);
    end

    % ===== Algorithm parameters =====
    l0         = max(5, floor(0.02 * n));  % initial rank estimate
    lcap       = Inf;                       % rank cap (set to e.g. floor(0.5*n) to limit)
    maxit_lsqr = 500;                       % max LSQR iterations
    blengamma  = 1.2;                       % Blendenpik oversampling factor

    fprintf("mu=%.2e, errtol=%.2e, l0=%d\n", mu, errtol, l0);
    fprintf("sv1(A)=%.2e, cond(A)=%.2e\n", dataSA(1), dataSA(1)/dataSA(end));

    % ===== Output directory =====
    OUTDIR = sprintf('./results/%s/%s', subdir, datasetname);

    fprintf("Running algorithms ...\n");

    % ===== Run algorithms =====
    % Uncomment/comment individual blocks to select which algorithms to run.

    % --- LSQR (baseline) ---
    run_lsqr('dataA', dataA, 'dataSA', dataSA, 'datab', datab, 'SEED', 11, ...
        'MU', mu, 'maxit', maxit_lsqr, 'OUTDIR', sprintf('%s/', OUTDIR));

    % --- NPCG ---
    try
        run_npcg('dataA', dataA, 'dataSA', dataSA, 'datab', datab, 'SEED', 11, ...
            'MU', mu, 'ERRTOL', errtol, 'maxit', maxit_lsqr, 'L0', l0, 'LCAP', lcap, ...
            'ALGNAME', 'npcg', 'OUTDIR', sprintf('%s/', OUTDIR));
    catch
        disp('NPCG failed');
    end

    % --- APLICUR variants ---
    run_aplicur('dataA', dataA, 'dataSA', dataSA, 'datab', datab, 'SEED', 11, ...
        'MU', mu, 'ERRTOL', errtol, 'maxit', maxit_lsqr, 'L0', l0, 'LCAP', lcap, ...
        'ALGNAME', 'aplicur', 'OUTDIR', sprintf('%s/', OUTDIR));

    run_aplicur('dataA', dataA, 'dataSA', dataSA, 'datab', datab, 'SEED', 11, ...
        'MU', mu, 'ERRTOL', errtol, 'maxit', maxit_lsqr, 'L0', l0, 'LCAP', lcap, ...
        'ALGNAME', 'aplicur_svdfree', 'OUTDIR', sprintf('%s/', OUTDIR));

    % % Other APLICUR variants for numerical studies (uncomment to run):
    % run_aplicur(..., 'ALGNAME', 'aplicur_singleshot',  ...);
    % run_aplicur(..., 'ALGNAME', 'aplicur_fixed_under', ...);
    % run_aplicur(..., 'ALGNAME', 'aplicur_fixed_over',  ...);
    % run_aplicur(..., 'ALGNAME', 'aplicur_fixed',       ...);
    % run_aplicur(..., 'ALGNAME', 'aplicur_under',       ...);
    % run_aplicur(..., 'ALGNAME', 'aplicur_over',        ...);
    % run_aplicur(..., 'ALGNAME', 'aplicur_correct',     ...);

    % --- Blendenpik ---
    run_bdp('dataA', dataA, 'dataSA', dataSA, 'datab', datab, 'SEED', 11, ...
        'MU', mu, 'maxit', maxit_lsqr, ...
        'ALGNAME', 'bdp', 'BLENGAMMA', blengamma, 'OUTDIR', sprintf('%s/', OUTDIR));

    % % Other Blendenpik variants (uncomment to run):
    % run_bdp(..., 'ALGNAME', 'bdp_sparse',     ...);
    % run_bdp(..., 'ALGNAME', 'bdp',            ...);
    % run_bdp(..., 'ALGNAME', 'bdp_sparse_isp', ...);

    % ===== Save basic output (problem info + optimal residuals) =====
    Amu = [dataA; mu * eye(n)];
    bmu = [datab; zeros(n, 1)];
    xmu_ast = lsqminnorm(Amu, bmu);

    output.mu       = mu;
    output.x_asts   = norm(datax);
    output.m        = m;
    output.n        = n;
    output.sparsity = nnz(dataA) / (m * n);
    output.svdAs    = dataSA;
    output.bs       = norm(bmu);
    output.relress  = norm(bmu - Amu * xmu_ast) / norm(bmu);
    if mu > 0
        output.relress_woreg = norm(datab - dataA * xmu_ast) / norm(datab);
        fprintf('Optimal rel. residual norm(b - A*xmu)        = %.4e\n', output.relress_woreg);
    end
    fprintf('Optimal rel. residual norm(bmu - Amu*xmu)/norm(bmu) = %.4e\n', output.relress);

    if ~exist(OUTDIR, 'dir'), mkdir(OUTDIR); end
    save(sprintf('./%s/output_basic.mat', OUTDIR), 'output', '-v7.3');

    fprintf("\nAll jobs done.\n");
end