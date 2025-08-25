function [error, approx] = approx_sin(x, num)
    approx = 0;
    for j = 0:num-1
        approx = approx + ((-1)^j* x^(2*j+1))/factorial(2*j+1);       
    end
    error = abs(approx - sin(x));
end