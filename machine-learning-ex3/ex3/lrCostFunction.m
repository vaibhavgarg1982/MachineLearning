function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

h_theta = sigmoid(X*theta);


temp2 = log(h_theta);
term1 = -y .* temp2; % term 1


temp4 = log(1-h_theta);
term2 = (1-y).* temp4; %term2

sum_term = sum(term1-term2);

J = sum_term ./m;
a = size(theta);


temp11 = lambda/(2*m)*theta;
J = J + lambda/(2*m)*sum(theta(2:a).^2);

%%%%%%%%%%%%

temp10 = X' * (h_theta - y);
grad_sum = (temp10) ./m;
grad = grad_sum;

grad(2:a) = grad(2:a) + lambda/m .*theta(2:a);



% =============================================================

grad = grad(:);

end
