%%% Project: NF-NARMAX
% Baseline ARMAX model
% Ref: https://nl.mathworks.com/help/ident/ref/armax.html
%
% Author: Wouter Kouw
% Last update: 18-01-2022

close all;
clear all;

%% Import data from julia notebook

load('data/train_data.mat')
load('data/validation_data.mat')

z_trn = iddata(y_trn', u_trn(1,:)', 1.0);
z_val = iddata(y_val', u_val(1,:)', 1.0);

%% Baseline ARMAX

% Orders
na = 2;
nb = 2;
nc = 2;
nk = 1;

% Define model
opt = armaxOptions;
opt.Focus = 'prediction';
opt.SearchMethod = 'lm';
opt.SearchOptions.MaxIterations = 100;
opt.Display = 'on';
sys = armax(z_trn, [na nb nc nk], opt);

% Visualize fit to training data
compare(z_trn, sys, 1)

% Validate performance
opt = predictOptions('InitialCondition','z');
pred_data = predict(sys, z_val, 1, opt);
predictions = pred_data.OutputData;
MSE = mean((y_val' - predictions).^2)
save('models/ARMAX222.mat', 'sys', 'predictions', 'MSE');

% Plot results
figure; hold on
N_val = length(y_val);
plot(1:N_val, y_val, 'LineWidth', 1, 'Color', 'black')
plot(1:N_val, predictions.OutputData, 'LineWidth', 3, 'Color', 'blue')
legend({'validation data', '1-step preds'})
xlim([0,N_val])
title(['ARMAX MSE = ' num2str(MSE)])
set(gcf, 'Color', 'w')
exportgraphics(gcf,'figures/ARMAX.png','Resolution',300)


