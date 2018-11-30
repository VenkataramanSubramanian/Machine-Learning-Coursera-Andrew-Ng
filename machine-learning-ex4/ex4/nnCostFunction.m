function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

hidden_layer_output = X * Theta1';
hidden_layer_output=sigmoid(hidden_layer_output);
hidden_layer_output = [ones(m,1) hidden_layer_output];
output_layer = hidden_layer_output * Theta2';

out = sigmoid(output_layer);

one_hot_y = eye(max(y));
one_hot_y = one_hot_y(y,:);

J = sum(sum((-1 .* one_hot_y .* log(out)) - ((1 .- one_hot_y) .* log(1 .- out)),2),1) / m;

if lambda!=0,
  J =J + (((sum(sum(Theta1(:,2:end).*Theta1(:,2:end),2),1) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end),2),1)) * lambda) ./ (2 *m));
end


delta_3 = (out - one_hot_y);
delta_2 = delta_3 * Theta2 .* hidden_layer_output .* (1 - hidden_layer_output);

Theta1_grad = (delta_2(:,2:end)' * X) ./ m;
Theta2_grad = (delta_3' * hidden_layer_output) ./ m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);

grad = [Theta1_grad(:) ; Theta2_grad(:)];

