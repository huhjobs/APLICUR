% demo.m — minimal working example for APLICUR
% Generates a synthetic least squares problem and solves it with APLICUR.
% See README.md for instructions on running full benchmarking experiments.

clear; clc;
addpath('./algorithms/', './utils/', './precondition/');

% ===== Generate synthetic problem =====
rng(42);
m = 3000; n = 500;
kappa = 1e13; SA = 10^2 * kappa.^(-linspace(0, 1, n).^5);
A = orth(randn(m, n)).* SA * orth(randn(n,n))';
x_true = randn(n, 1);
b = A * x_true + 0.01 * randn(m, 1);

mu = 1e-4;  % regularization parameter (set to 0 for unregularized)

Amu = [A; mu * eye(n)];
bmu = [b; zeros(n, 1)];

% ===== Algorithm parameters =====
block_size = max(5, floor(0.02 * n));   % initial rank / block size
err_tol    = 30 * mu;                   % CUR approximation error tolerance
max_it     = floor(n/block_size);       % max CUR refinement rounds

params_lsqr.tol   = 1e-10;
params_lsqr.maxit = 500;

ns_var.sketch  = 'SparseEmbedding';     % sketch type: 'SparseEmbedding', 'Gaussian', 'SRFT'
ns_var.svdfree = false;                 % set true for SVD-free variant

% ===== Run APLICUR =====
[xs, ts_xs, resvecs, ts_resvecs, timings, timestamps, ls, flag, ~] = ...
    aplicur(A, mu, Amu, bmu, block_size, err_tol, max_it, params_lsqr, true, ns_var);

% ===== Display results =====
x_sol = xs(:, end);
fprintf("\nFinal relative residual: %.4e\n", norm(A*x_sol - b) / norm(b));
fprintf("CUR preconditioner rank: %d\n", ls);
fprintf("Total time             : %.3f sec\n", timings.timing_aplicurprec + timings.timing_aplicurlsqr);

% ===== Plot convergence =====
b_norm = norm(b);
marker_idx = round(linspace(1, length(resvecs), 20));

figure('Units', 'inches', 'Position', [1 1 10 4]);

% --- Residual vs iterations ---
subplot(1, 2, 1);
semilogy(resvecs / b_norm, '-o', 'MarkerIndices', marker_idx, 'MarkerSize', 4);
xlabel('LSQR iterations'); ylabel('Rel. residual');
title('Convergence vs iterations'); grid on;

% --- Residual vs time ---
subplot(1, 2, 2);
semilogy(timestamps, ts_resvecs / b_norm, '-o', 'MarkerIndices', marker_idx, 'MarkerSize', 4);
xlabel('Time (s)'); ylabel('Rel. residual');
title('Convergence vs time'); grid on;

sgtitle('APLICUR convergence');