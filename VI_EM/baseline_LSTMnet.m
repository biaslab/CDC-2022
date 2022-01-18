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
load('validation_data.mat')

%% Baseline LSTM net

% Define network architecture
numResponses = 1;
featureDimension = 2;
numHiddenUnits = 100;
maxEpochs = 500;
miniBatchSize = 10;
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
    'ValidationData',[{u_val} {y_val}]);

% Train using parallel threads
poolobj = parpool;
fullNet = trainNetwork(u_trn, y_trn, Networklayers, options);
delete(poolobj)

% Validate performance
predictions = predict(fullNet, u_val);
MSE = mean((y_val - predictions).^2)
save('LSTMnet.mat', 'fullNet', 'predictions', 'MSE');

% Plot results
figure; hold on
N_val = length(y_val);
plot(1:N_val, y_val, 'LineWidth', 1, 'Color', 'black')
plot(1:N_val, predictions', 'LineWidth', 3, 'Color', 'blue')
legend({'validation output', '1-step preds'})
xlim([0,N_val])
title(['LSTM-net: [u_t u_{t-1}] -> y_t' newline 'MSE = ' num2str(MSE)])
set(gcf, 'Color', 'w')
exportgraphics(gcf,'LSTMnet.png','Resolution',300)


