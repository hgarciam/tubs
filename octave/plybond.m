%% Initialization
clear ; close all; clc

fprintf('//// Plybond analysis ////\n');
fprintf('//// Hector Garcia Morales 2018 ////\n');

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% load data
data = load('plybond.dat');
X = data(:,1:4);
%X = data(:, 2);
y = data(:, 5);
m = length(y); % number of training examples
n = length(X(1,:)); % number of training examples
fprintf('Number of samples = %i\n', m);
fprintf('Number of features = %i\n', n);

%% Setup the parameters you will use for this exercise
input_layer_size = 4;  % 20x20 Input Images of Digits
hidden_layer_size = 5;   % 25 hidden units
num_labels = 1;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% Add ones column to X
%X = [ones(m, 1), data(:,1:4)];
%X = [ones(m, 1), data(:,1:4)];

%fprintf('Min plybond = %f\n', min(data(:,1)));
%fprintf('Max plybond = %f\n', max(data(:,1)));
%fprintf('Mean plybond = %f\n', mean(data(:,1)));
%
%fprintf('Min intD = %f\n', min(data(:,3)));
%fprintf('Max intD = %f\n', max(data(:,3)));
%fprintf('Mean intD = %f\n', mean(data(:,3)));
%
%fprintf('Min wall = %f\n', min(data(:,4)));
%fprintf('Max wall = %f\n', max(data(:,4)));
%fprintf('Mean wall = %f\n', mean(data(:,4)));
%
%fprintf('Min hum = %f\n', min(data(:,5)));
%fprintf('Max hum = %f\n', max(data(:,5)));
%fprintf('Mean hum = %f\n', mean(data(:,5)));

%plybond = X(:,2);
%intD = X(:,3);
%wall = X(:,4);
%hum = X(:,5);

% PLot data

%plot(X(:,2:end),y,'o')
%xlabel('X')
%label('Y')
%title('Overview')

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f '], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 100);

%  You should also try different values of lambda
lambda = 1.0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);


