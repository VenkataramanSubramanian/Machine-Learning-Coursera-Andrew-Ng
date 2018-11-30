function [X_poly] = polyFeatures(X, p)


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

for i=1:p,
  X_poly(:,i) = X .^ i;
end

end
