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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


[nbr, nbc] = size(X);

%2*m
m2 = nbr*2;
%lambda / 2*m
lambda_m2 = lambda/m2;

%compute h(x)
h = X*theta;
%compute (h(x) - y)^2
sqrt_h = (h.*h).-((h.*2).*y).+(y.*y);
%compute 1/2m sum ( *(h(x) - y)^2 )
sum_h = sum(sqrt_h)/m2;
%compute theta^2
sqrt_theta = theta.*theta;
%compute sum from 2 to end sqrt_theta
sum_sqrt_theta = sum(sqrt_theta(2:end))*lambda_m2;
J = sum_h + sum_sqrt_theta;



lambda_m = lambda/nbr;
%gradient for theta 0
% sum (h*x - y*x) / m
grad(1) = sum((h.*X(:,1)).-(y.*X(:,1)))/nbr;
%gradient for thetat 1:end

[nbr_theta, nbc_theta] = size(theta); 
for i = 2:(nbr_theta)
	grad(i) = sum((h.*X(:,i)).-(y.*X(:,i)))/nbr + (theta(i)*lambda_m);
endfor


% =========================================================================


end
