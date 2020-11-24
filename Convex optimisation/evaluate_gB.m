function gB = evaluate_gB(beta_vect, X, y, n, m, dim, lambda, eval, norm)
% Evaluate the objective function & gradient for MNIST classification
%
% Ruth Misener, 01 Feb 2016
%
% INPUTS: beta_vect: Domain point to evaluate (matrix dimension: n x dim)
%                    On entry beta_vect is stored as a vector, but it is
%                    equivalent to a n x dim matrix.
%         lambda:    Regularization parameter (scalar)
%         X:         Input features (matrix dimension: m x n)
%         y:         Output label (vector length: m)
%         n:         Input features
%         m:         Test cases
%         dim:       Number of digits (10 digits, 0 - 9)
%         eval:      Eval function or gradient? 0: Fcn, 1: Gradient
%         norm:      Use 1-norm or (2-norm)^2? 1: 1-norm, 2: (2-norm)^2

beta_matrix = zeros(n, dim);
for k = 1:dim
    beta_matrix(:,k) = beta_vect(((k - 1)*n + 1):(k*n));
end

if eval == 0          % Evaluate the function
    gB = 0;
    for i = 1:m       % Loop over all the test cases    
        gB_inner = 0;
        
        for k = 1:dim % Loop over all the digits
             gB_inner = gB_inner + exp( X(i,:) * beta_matrix(:,k) );            
        end
        
        gB = gB + (log(gB_inner)) - X(i,:) * beta_matrix(:,y(i) + 1);
    end
    
    for k = 1:dim   % Loop over all the digits
        if (norm == 2)
            gB = gB + (lambda.*dot(beta_matrix(:,k), beta_matrix(:,k)));  
        else
            gB = gB + (lambda.*sum(abs(beta_matrix(:,k)))); 
        end
    end
    
else                % Evaluate the derivative

    Y = zeros(m, dim);
    Z = zeros(m, m);   
    
    for i = 1:m     % Loop over all the test cases  
         Y(i, y(i) + 1) = 1;
   
        for k = 1:dim % Loop over all the digits
            Z(i,i) = Z(i,i) + exp( X(i,:) * beta_matrix(:,k) );
        end
        Z(i,i) = 1/Z(i,i);
    end
    
    Y = sparse(Y);
    Z = sparse(Z);
  
    gB = (X' * ( Z * exp(X * beta_matrix) - Y));
    
    if (norm == 2)
        gB = gB + 2 .* lambda .* beta_matrix;
    end
    
    for k = 1:dim
        gB_vect((k - 1)*n + 1 : k*n) = gB(:,k);
    end
        
    gB = gB_vect;
    
end
end

