function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

dJthetadTheta0 = 0;
dJthetadTheta1 = 0;

for i = 1:m 

	hypothesis = theta(1,1)*X(i,1) + theta(2,1)*X(i,2);  %h_theta(x) = theta0 + theta1*x
	calculatedError = (hypothesis- y(i,1));

	dJthetadTheta0 += (calculatedError*X(i,1))/m;

	dJthetadTheta1 += (calculatedError*X(i,2))/m;
end


	theta(1,1) = theta(1,1) - (alpha*dJthetadTheta0);

	theta(2,1) = theta(2,1) - (alpha*dJthetadTheta1);
	
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
