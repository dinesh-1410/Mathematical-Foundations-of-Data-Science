function pi_approx = compute_pi()
    % Number of terms in the series
    n = 10000;
    
    % Initialize the sum
    sum = 0;
    
    % Compute the sum of the series
    for k = 0:n
        term = ((-1)^(k) )/ (2*k + 1);
        sum = sum + term;
    end
    
    % Multiply by 4 to get the approximation of pi
    pi_approx = 4 * sum
end
