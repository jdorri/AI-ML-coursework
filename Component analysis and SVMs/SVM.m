function SVM()

    % Solve the quadratic optimisation problem. Estimate the labels for 
    % each of the test samples and report the accuracy of your trained SVM 
    % utilising the ground truth labels for the test data.

    load('X.mat'); 
    load('l.mat');
    load('X_test.mat');
    load('l_test.mat');    
    
    % prep 
    [~, n] = size(X); 
    S_t = 1 / n * (X * X'); % compute covariance matrix

    % enter coefficent matrices
    H = (l * l') .* (X' * inv(S_t) * X);
    H = (H + H') / 2;
    f = -ones(n, 1)';
    A = zeros(1, n);
    c = 0;
    A_e = l';
    c_e = 0;
    a_1 = zeros(n, 1)';
    a_u = ones(n, 1);
    
    % perform optimisation
    a = quadprog(H, f, A, c, A_e, c_e, a_1, a_u); 
 
    
    % compute optimal weight vector 
    w = inv(S_t) * X * (a .* l);
  
    % compute optimal b
    threshold = 10^-5;
    k = find(a > threshold); % compute vector of indexes corresponding to nonzero multipliers
    [n, ~] = size(k); % store number of support vectors
    b = (1 / n) * sum(l(k) - (X(:, k)' * w)); 
    
    % compute predictions using decision function
    [~, n] = size(X_test);
    l_pred = zeros(n, 1);
    l_raw = (w' * X_test) + b;
    for i = 1:n
        if l_raw(i) > 0 
            l_pred(i) = 1;
        else l_pred(i) = -1;
        end
    end
    
    % calculate accuracy 
    correct = zeros(n, 1);
    for i = 1:n
        if l_pred(i) == l_test(i)
            correct(i) = 1;
        end
    end   
    
    accuracy = sum(correct) / n; 
    fprintf('Accuracy on the test set is %3.2f\n', accuracy);

end