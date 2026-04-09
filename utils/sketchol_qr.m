function [Q,R] = sketchol_qr(M,needQ,needR)
% https://www.ethanepperly.com/index.php/2024/06/25/neat-randomized-algorithms-randomized-cholesky-qr/

    if nargin < 2, needQ = true; end
    if nargin < 3, needR = true; end
       
    S = sparsesign(2*size(M,2),size(M,1),8); % Sparse Embedding
    R_sket   = qr(full(S*M),'econ');
    Q_approx = M/R_sket;
    
    R_chol = chol(full(Q_approx'*Q_approx));
    if needQ, Q = Q_approx/R_chol;
        else, Q = []; end
    if needR, R = R_chol * R_sket;
        else, R = []; end
end