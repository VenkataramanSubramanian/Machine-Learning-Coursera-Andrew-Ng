function [J, grad] = lrCostFunction(theta, X, y, lambda)

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


val =X*theta;
val = sigmoid(val);

theta(1) = 0; 
J = (sum(y .* log(val) .+ ((1 .- y) .* log(1 .- val)))) ./ (-1 .* m) + ((lambda * sum( theta .^ 2)) / (2 * m));
add_val = (lambda .* theta) ./ m;
grad = ((X' * (val .- y))) ./ m .+ add_val; 

grad = grad(:);

end
