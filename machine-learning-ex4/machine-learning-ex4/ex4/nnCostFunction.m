function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X=[ones(m,1),X];
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a1=X;
z1=sigmoid(Theta1*a1');
a2=[ones(1,size(z1,2));z1];
z3=Theta2*a2;
a3=sigmoid(z3);
h=a3;

ymat=zeros(num_labels,m);

for i=1:num_labels,
  ymat(i,:) = (y==i);
endfor
J=sum(sum((-1*ymat.*log(h))-(1-ymat).*log(1-h)))/m;

theta1reg=Theta1(:,2:size(Theta1,2));
theta2reg=Theta2(:,2:size(Theta2,2));

reg=(lambda/(2*m))*sum(sum(sum(theta1reg.^2))+sum(sum(theta2reg.^2)));

J=J+reg;

%d3=a3-y;
%d2=(Theta1'*d3).*[ones(1,m);sigmoidgradient(z2)];
%Theta2_grad = (1/m) * d3 * a2';
%Theta1_grad = (1/m) * d2(2:end, :) * a1';
%d1=a1.*(1-a1);
%Theta2_grad=Theta2_grad+(lambda/m)*([zeros(size(Theta2,1),1),Theta2(2:end)]);
%Theta1_grad=Theta1_grad+(lambda/m)*([zeros(size(Theta1,1),1),Theta1(2:end)]);
%grad=[Theta1_grad(:);Theta2_grad(:)];


for t=1:m,
  a1=X(t,:);
  z2=Theta1*a1';
  a2=sigmoid(z2);
  a2=[1;a2];
  a3=sigmoid(Theta2*a2);
  
  d3=a3-ymat(:,t);
  z2=[1;z2];
  d2=(Theta2'*d3).*sigmoidGradient(z2);
  
  d2=d2(2:end,:);
  
  Theta2_grad=Theta2_grad+(d3*a2');
  Theta1_grad=Theta1_grad+(d2*a1);

endfor

%Theta1_grad=Theta1_grad./m;
%Theta2_grad=Theta2_grad./m;

Theta1_grad(:,1)=Theta1_grad(:,1)./m;
Theta2_grad(:,1)=Theta2_grad(:,1)./m;

%When j=0
Theta1_grad(:,2:end) = Theta1_grad(:,2:end)./m + ( (lambda/m) * Theta1(:,2:end) );
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)./m + ( (lambda/m) * Theta2(:,2:end) );

grad=[Theta1_grad(:);Theta2_grad(:)];














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
end