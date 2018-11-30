function [J, grad] = linearRegCostFunction(X, y, theta, lambda)


% Initialize some useful values
m = length(y); % number of training examples


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = ( sum((((X * theta) .- y) .^ 2),1) + (lambda .* sum((theta(2:end) .^ 2),1)) ) ./ (2 * m);

grad =  (X' * ((X * theta ) .- y)) / m;

grad(2:end) = grad(2:end) + ( (lambda / m) .* theta(2:end) );

grad = grad(:);

end	
