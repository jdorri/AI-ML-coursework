function U = LDA(data, labels)
% Transpose data 
rawX = data';

% compute matrix M ontaining in all columns the mean 
% center data 
[~, n] = size(rawX);
onesArray = ones(n, 1);
M = rawX * ((onesArray * onesArray') / n);
X = rawX - M;

% compute summuary of unique labels and their count
% shows there are 68 unique lables, all with count = 5
[C, ~, ic] = unique(labels);
label_counts = accumarray(ic, 1);
value_counts = [C, label_counts];

% compute E 
elem = 1/5;
rowE = [elem elem elem elem elem];
E = [rowE; rowE; rowE; rowE; rowE];

% compute M, compute I
M = kron(eye(68), E);
I = eye(340);

% compute mean-centered dot product matrix A for eigenvalue analysis
A = (I - M) * (X' * X) * (I - M);

% enforce symmetry
A = (A' + A) / 2;

% diagonalise A
% store k positive eigenvalues in S_w, where k = N - (C + 1)
[~, n] = size(A);
k = n - (68 + 1);
[V_w, S_w] = eigs(A, k);

% compute eigenvectors of A (stored in U)
U = X * (I - M) * V_w * inv(S_w);

% compute matrix X_b of projected class means 
X_b = U' * X * M;

% enforce symmetry 
C =  X_b * X_b'; 
C = (C' + C) / 2;

% perform eigenvalue analysis on X_b * X_b' to find Q
% keeping only at most (68 - 1) positive eigenvalues 
[Q, ~] = eigs(C, 67);

% transform U 
U = U * Q;


end