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

%% Vector implementaton
z = theta' * X'; %% size(z): 1x118; size(theta): 3x1; size(X): 118x3 

hypothesis = sigmoid(z);	%%size(hypothesis): 1x118 

%%j = -(y*log(hypothesis) + (1 - y)*log(1 - (hypothesis)))/m;	%%size(j): 118x118
j = -(y'*log(hypothesis') + (1 - y')*log(1 - (hypothesis')))/m;

Error = hypothesis - y';	%%for gradient calculation; size(Error): 1x118; size(y): 118x1

J1 = 0;
J2 = 0;
%%for i = 1:m

%%cost calculation --->

%%	J1 += j(i); 

%%gradient calculation --->


%	grad(1,1) += (Error(i)*X(i,1))/m;
%	grad(2,1) += (Error(i)*X(i,2))/m;
%	grad(3,1) += (Error(i)*X(i,3))/m;

%%end

	J1 = j;

	theta(1,1) = 0;

	J2 = (theta'*theta)*(lambda/(2*m));

	J = J1 + J2;

%%	J = j;

%%gradient calculation --->
%%	grad = X' * (Error);
%%	grad =  ((Error)*X)'/m;	

	grad1 =  ((Error)*X)'/m;	

	grad2 = theta*(lambda/m);

	grad = grad1 + grad2;

%% non-vector implementation

%J1 = 0;
%J2 = 0;

%for i = 1:m 
	
%	z = theta(1,1)*X(i,1) + theta(2,1)*X(i,2) + theta(3,1)*X(i,3);
	
%	hypothesis = sigmoid(z);



%% cost calculation --->
%	J1 += -(y(i)*log(hypothesis) +  (1-y(i))*log(1-(hypothesis)) )/m;


%% gradient calculation --->

%	Error = hypothesis - y(i);

%	grad(1,1) += (Error*X(i,1))/m;

%	grad(2,1) += (Error*X(i,2))/m;

%	grad(3,1) += (Error*X(i,3))/m;


%end

%cost --->

%	J2 = (((theta(2,1))^2 + (theta(3,1))^2)*lambda)/(2*m);

%	J = J1 + J2;

%gradient --->

%	grad(2,1) += (theta(2,1)*lambda)/m; 

%	grad(3,1) += (theta(3,1)*lambda)/m; 

% =============================================================

end
