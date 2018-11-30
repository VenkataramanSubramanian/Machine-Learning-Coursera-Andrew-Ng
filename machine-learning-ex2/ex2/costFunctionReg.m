function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

[J1,grad1] = costFunction(theta, X, y);
theta(1) = 0; 
J = J1 + ((lambda * sum( theta .^ 2)) / (2 * m));
add_val = (lambda .* theta) ./ m;
grad = grad1 .+ add_val; 

end
