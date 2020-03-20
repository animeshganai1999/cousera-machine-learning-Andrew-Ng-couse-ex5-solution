function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
disp(m);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Compute cost
hypothesis = X*theta;
val = (hypothesis - y);
J = (val'*val)./(2*m) + (lambda.*(sum(theta(2:end,:).^2,1)))./(2*m);
%theta(1) should not regularized

%gradient
new_theta = [zeros(1,size(theta,2));theta(2:end,:)]; %convert 1st row of theta to '0' for each column ,so that we can compute grad in one step

grad = (sum(val.*X,1))'./m + (lambda.*new_theta)./m;
%sum(val.*X,1) is a 1x2 matrix but new_theta is 2x1 matrix so we transpose it to make grad 2x1

% =========================================================================

grad = grad(:);

end
