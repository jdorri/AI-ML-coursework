function ReturnVal = SolveMNIST_Gradient(tol, num_iter, step_size, ...
                                         lambda)
% Build a classifier for recognising hand-written digits from images
%
% Ruth Misener, 01 Feb 2016
%
% INPUTS: tol:       Optimality tolerance; check if algorithm converged
%         num_iter:  Maximum number of iterations
%         step_size: Step size
%         lambda:    Regularisation parameter

% Initialise the training set --------------------------------------------
load mnist.mat
n   = 1000; % Input features
m   = 1000; % Test cases
dim =   10;

% l-2 Regulariser
norm_type = 2;

% Initialise a starting point for the algorithm --------------------------
beta_guess = zeros(1,n*dim);

beta_eval  = evaluate_gB(beta_guess, X, y, n, m, dim, lambda, ...
                         0, norm_type);
beta_grad  = evaluate_gB(beta_guess, X, y, n, m, dim, lambda, ...
                         1, norm_type);

% Store beta guesses at each iteration
beta_guess_iter(1,:) = beta_guess; 

% Store the function value at each iteration
fcn_val_iter(1)      = beta_eval;  

fprintf('\niter=%d; Func Val=%f; FONC Residual=%f',...
        0, beta_eval, norm(beta_grad));

% Iterative algorithm begins ---------------------------------------------
for i = 1:num_iter                        
 
    % Step for gradient descent ------------------------------------------
    % *** Insert gradient descent code here ***
   
    % perform update with CONSTANT step
    % beta_guess = beta_guess - step_size*beta_grad;
    
    % perform update with ADAPTIVE step
    d = beta_grad; % search direction
    adapt_step_size = backtracking(beta_guess, d, lambda); % compute step-size
    beta_guess = beta_guess - adapt_step_size*beta_grad; 
     
    % Update with the new iteration --------------------------------------
    beta_guess_iter(i+1,:) = beta_guess;
    
    beta_eval              = evaluate_gB(beta_guess, X, y, n, m, dim, ...
                                         lambda, 0, norm_type);
                                     
    fcn_val_iter(i+1)      = beta_eval;
    
    beta_grad              = evaluate_gB(beta_guess, X, y, n, m, dim, ...
                                         lambda, 1, norm_type);
                         
    % Check if it's time to terminate ------------------------------------

    % Check the FONC?
    % Store the norm of the gradient at each iteration
    convgsd(i) = norm(beta_grad); % <-- Correct this!!
    
    % Check that the vector is changing from iteration to iteration?
    % Stores length of the difference between the current beta and the 
    % previous one at each iteration
    lenXsd(i)  = norm(beta_guess_iter(i+1,:)-beta_guess_iter(i,:)); % <-- Correct this!!
    
    % Check that the objective is changing from iteration to iteration?
    % Stores the absolute value of the difference between the current 
    % function value and the previous one at each iteration
    diffFsd(i) = abs(fcn_val_iter(i+1)-fcn_val_iter(i)); % <-- Correct this!!
    
    fprintf('\niter=%d; Func Val=%f; FONC Residual=%f; Sqr Diff=%f',...
            i, beta_eval, convgsd(i), lenXsd(i));
    
    % Check the convergence criteria?
    if (convgsd(i) <= tol)
        fprintf('\nFirst-Order Optimality Condition met\n');
        break; 
    elseif (lenXsd(i) <= tol)
        fprintf('\nExit: Design not changing\n');
        break;
    elseif (diffFsd(i) <= tol)
        fprintf('\nExit: Objective not changing\n');
        break;
    elseif (i + 1 >= num_iter)
        fprintf('\nExit: Done iterating\n');
        break;
    end
    
end

ReturnVal = beta_guess;

% plotting
number_of_iters = length(fcn_val_iter);
iters_array = [1:number_of_iters];
semilogy(iters_array, fcn_val_iter);

end