function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

values = [0.01 0.03 0.1 0.3 1.3 10 30];

try_size=length(values);
flag1=0;
min_err=0;

for i=1:try_size,
  for j=1:try_size,
    model = svmTrain(X, y, values(i), @(x1, x2) gaussianKernel(x1, x2, values(j)));
    pred = svmPredict(model,Xval);
    err= mean(double(pred ~= yval));
    if(flag1==0),
      min_err=err;
      flag1=1;
    end
    if(err<=min_err),
      C=values(i);
      sigma=values(j);
      min_err=err;
    end
  end
end



end
