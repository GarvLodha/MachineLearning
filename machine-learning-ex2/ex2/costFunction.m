function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
% Note: grad should have the same dimensions as theta
%

%% Vector implementaton
z = theta' * X'; %% size(z): 1x100; size(theta): 3x1; size(X): 100x3 

hypothesis = sigmoid(z);	%%size(hypothesis): 1x100

%j = -(y*log(hypothesis) + (1 - y)*log(1 - (hypothesis)))/m;
j = -(y'*log(hypothesis') + (1 - y')*log(1 - (hypothesis')))/m;

Error = hypothesis - y';	%%for gradient calculation; size(Error): 1x100; size(y): 100x1

%for i = 1:m

%%cost calculation --->

%	J += j(i); 

%%gradient calculation --->


%	grad(1,1) += (Error(i)*X(i,1))/m;
%	grad(2,1) += (Error(i)*X(i,2))/m;
%	grad(3,1) += (Error(i)*X(i,3))/m;

%end
	J = j;
%%gradient calculation --->
%%	grad = X' * (Error);
	grad =  ((Error)*X)'/m;	


%% non-vector implementation


%for i = 1:m 
	
%	z = theta(1,1)*X(i,1) + theta(2,1)*X(i,2) + theta(3,1)*X(i,3);
	
%	hypothesis = sigmoid(z);

%% cost calculation --->
%	J += -(y(i)*log(hypothesis) +  (1-y(i))*log(1-(hypothesis)) )/m;

%% gradient calculation --->

%	Error = hypothesis - y(i);

%	grad(1,1) += (Error*X(i,1))/m;

%	grad(2,1) += (Error*X(i,2))/m;

%	grad(3,1) += (Error*X(i,3))/m;


%end


end
