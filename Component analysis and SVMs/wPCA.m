function U = wPCA(data)
% Transpose data 
rawX = data';

% compute matrix M containing in all columns the mean 
[~, n] = size(rawX);
onesArray = ones(n, 1);
M = rawX * ((onesArray * onesArray') / n);

% center data 
X = rawX - M;

% compute dot product matrix for eigenvalue analysis
A = X' * X;

% diagonalise A
% store k largest eigenvalues in D
[~, n] = size(A);
k = n - 1;
[V, D] = eigs(A, k);

% compute eigenvectors of A (stored in U)
U = X * V * inv(sqrtm(D));

% perform whitening transformation
U = U * inv(sqrtm(D));


end