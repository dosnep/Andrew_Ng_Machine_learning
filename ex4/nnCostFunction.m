function [J grad] = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


%Construction du nouveau test (version binaire)


newY = zeros(size(y),num_labels);
for i = 1:size(y)
	newY(i,y(i)) = 1;
endfor

[nbr, nbc] = size(X);

%Construction du layer1
a1 = [ones(nbr,1),X];
size(a1);

%Construction du layer2
a2 = a1*Theta1';
z2 = a2;
a2 = sigmoid(a2);
a2 = [ones(nbr,1), a2];
%z2 = [ones(nbr,1), z2];

%Construction du layer3
a3 = a2*Theta2';
a3 = sigmoid(a3);
size(a3);
output = a3;

%-y*log(h)
firstPart = -newY.*log(output);
%y*log(1-h)
secondPart = newY.*log(1-output);
%-log(1-h)
thirdPart = -log(1-output);
%-y*log(h)-(1-y)*log(1-h)
composite = firstPart.+secondPart.+thirdPart;

J = sum(sum(composite,2),1)./nbr;

%thetha1^2
sqThe1 = Theta1.*Theta1;
%thetha2^2
sqThe2 = Theta2.*Theta2;
%Sum row and columm theta1
sumThe1 = sum(sum(sqThe1(:,2:end),2),1); 
%Sum row and columm theta2
sumThe2 = sum(sum(sqThe2(:,2:end),2),1); 
%2*m
denum = 2*nbr;
%lambda/2*m
factor  = lambda/denum;
Regularized = (sumThe1*factor)+(sumThe2*factor);
J = J+Regularized;


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% a(3) -y
small_delta_3 = output.-newY; 
% small_delta_2 = theta2 * small_delta_2 * sigmoid_gradient(z)
small_delta_2 = (small_delta_3*Theta2(:,2:end)).*sigmoidGradient(z2);
%delta_2 = a_2' * small_delata_3
delta_2 = small_delta_3'*a2;
Theta2_grad = delta_2;
delta_1 = small_delta_2'*a1;
Theta1_grad = delta_1;

Theta1_grad = Theta1_grad./nbr;
Theta2_grad = Theta2_grad./nbr;



%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



[rowT1, colT1] = size(Theta1);
[rowT2, colT2] = size(Theta2);

for i = 1:rowT1
	for j = 2:colT1;
		Theta1_grad(i,j) = Theta1_grad(i,j)+((lambda*Theta1(i,j))/nbr);	
	endfor
endfor


for i = 1:rowT2
	for j = 2:colT2;
		Theta2_grad(i,j) = Theta2_grad(i,j)+((lambda*Theta2(i,j))/nbr);	
	endfor
endfor



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
