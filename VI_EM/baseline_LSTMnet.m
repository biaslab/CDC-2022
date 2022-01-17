%%% Project: NF-NARMAX
% Baseline LSTM network (designed for LTI system)
% Ref: https://nl.mathworks.com/help/ident/ug/use-lstm-for-linear-system-identification.html
%
% Author: Wouter Kouw
% Last update: 17-01-2022

close all;
clear all;

%% Import data from julia notebook

load('train_data.mat')
load('test_data.mat')

%% Baseline LSTM net

% Define network architecture
numResponses = 1;
featureDimension = 1;
numHiddenUnits = 100;
maxEpochs = 1000;
miniBatchSize = 200;
Networklayers = [sequenceInputLayer(featureDimension) ...
    lstmLayer(numHiddenUnits) ...
    fullyConnectedLayer(numResponses) ...
    regressionLayer];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',10, ...
    'Shuffle','once', ...
    'Plots','training-progress',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',100,...
    'Verbose',0,...
    'ValidationData',[{input_test'} {output_test'}]);

% Train using parallel threads
poolobj = parpool;
fullNet = trainNetwork(input_train', output_train', Networklayers, options);
delete(poolobj)
save('fullNet');

% Validate performance
predictions = predict(fullNet, input_test');
MSE = mean((output_test' - predictions).^2)

% Plot results
figure; hold on
N_val = length(output_test);
plot(1:N_val, output_test', 'LineWidth', 1, 'Color', 'black')
plot(1:N_val, predictions', 'LineWidth', 3, 'Color', 'blue')
legend({'test output', '1-step preds'})
xlim([0,N_val])


