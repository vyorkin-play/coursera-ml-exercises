function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    J = computeCost(X, y, theta);
    prevTheta = theta;

    theta(1) = theta(1) - (alpha / m) * sum(X * prevTheta - y);
    theta(2) = theta(2) - (alpha / m) * sum(X(:, 2)' * (X * prevTheta - y));

    J_history(iter) = J;
end

end
