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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

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

end
