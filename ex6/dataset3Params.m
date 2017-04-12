function [C, sigma] = dataset3Params(X, y, Xval, yval, x1, x2)
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

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_list = [0.01,0.03,0.03,0.1,0.3,1,3,10,30];
sigma_list = c_list;
min_score = 0;


for i = 1:9
	for j = 1:9

		current_c = c_list(i);
		current_sigma = sigma_list(j);
		model= svmTrain(X, y, current_c, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
		predictions = svmPredict(model, Xval);				
		current_score = mean(double(predictions ~= yval));
		if(i == 1 || min_score > current_score)
			min_score = current_score
			C = current_c;
			sigma = current_sigma;		
		endif;
		
	endfor
endfor


% =========================================================================

end
