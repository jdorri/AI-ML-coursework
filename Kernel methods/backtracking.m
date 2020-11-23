function ReturnVal = backtracking(x, d, lambda)

% as before 
load mnist.mat
n   = 1000; 
m   = 1000; 
dim =   10;
norm_type = 2;

% reset step-size to 1
a_k=1;
        
% backtracking parameters
c_alpha=0.5;
c_beta=1.2;

% loop through until Armijo's condition is met
while evaluate_gB(x-a_k*d, X, y, n, m, dim, lambda, 0, norm_type) ...
        > evaluate_gB(x, X, y, n, m, dim, lambda, 0, norm_type) - c_alpha*a_k*norm(d)^2
                     
    a_k = (1/c_beta)*a_k;
end

alpha = a_k;
ReturnVal = alpha;

end 
   