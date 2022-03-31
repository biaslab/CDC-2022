close all;
clear all;

addpath(genpath("../algorithms/ILS-estimator-NARMAX"));
addpath(genpath("../datasets"));

%%

% Series of train sizes
% trn_sizes = 2.^[6:10];
trn_sizes = [100,200,500,1000];
num_trnsizes = length(trn_sizes);

% Define transient and test indices
transient = 0;
ix_tst = [1:1000] + transient;

options.na = 1; % # output delays
options.nb = 1; % # input delays
options.ne = 1; % # innovation delays
options.nd = 3; % # degree polynomial nonlinearity

M_m = options.na + 1 + options.nb + options.ne;

options.N = 2^16;
options.P = 1;
options.M = 1;

options.fMin = 0;
options.fMax = 100;
options.fs = 1000;
options.type = 'odd';

options.stdu = 1.0;
options.stde = .05;

options.dc = false;
options.crossTerms = true;
options.noiseCrossTerms = false;
options.normalize = false;

% Number of repetitions
num_repeats = 100;

% Preallocate result arrays
results_prd = zeros(num_repeats, num_trnsizes);
results_sim = zeros(num_repeats, num_trnsizes);

r = 1;
while r <= num_repeats
    dbox = waitbar(r/num_repeats);
    
    % Generate signal
%     [yTrain, yTest, uTrain, uTest, eTrain, eTest, system] = gen_signal(options);
    
%     N_m = size(system.comb,2);
    
%     if (max(abs(yTrain)) < 100) && (sum(isnan(yTrain))==0)
    
        % Write signal to file
%         save("../datasets/verification/realizations/NARMAXsignal_stde"+num2str(options.stde)+"_degree"+num2str(options.nd)+"_delays"+num2str(M_m)+"_D"+num2str(N_m)+"_r" + string(r) + ".mat", "yTrain", "yTest", "uTrain", "uTest", "eTrain", "eTest", "system", "options")
        load("../datasets/verification/realizations/NARMAXsignal_stde"+num2str(options.stde)+"_degree"+num2str(options.nd)+"_delays"+num2str(M_m)+"_D"+num2str(22)+"_r" + string(r) + ".mat", "yTrain", "yTest", "uTrain", "uTest", "eTrain", "eTest", "system", "options")
        % Preallocate result arrays
        RMS_prd = zeros(1,num_trnsizes);
        RMS_sim = zeros(1,num_trnsizes);
        
        for n = 1:num_trnsizes
            
            % Establish length of training signal
            ix_trn = [1:trn_sizes(n)] + transient;
        
            % Slice data
            dataTrain.u = uTrain(ix_trn);
            dataTrain.y = yTrain(ix_trn);
            dataTest.u = uTest(ix_tst);
            dataTest.y = yTest(ix_tst);

            % ILS estimator
            [modelNarmaxIter,eNarmaxIter] = fEstPolNarmax(dataTrain,options);

            % 1-step ahead prediction
            [yPredIterTest,ePredIterTest] = fPredPolNarmax(dataTest,modelNarmaxIter);

            % Simulation
            ySimIterTest = fSimPolNarmax(dataTest,modelNarmaxIter);

            % Compute RMS
%             RMS_prd(n) = rms(dataTest.y - yPredIterTest);
%             RMS_sim(n) = rms(dataTest.y - ySimIterTest);
            
            RMS_prd(n) = sqrt(mean((dataTest.y - yPredIterTest).^2));
            RMS_sim(n) = sqrt(mean((dataTest.y - ySimIterTest).^2));
            
        end
        
        % Write results to file
        save("results/results-NARMAX_ILS_stde"+num2str(options.stde)+"_pol"+num2str(options.nd)+"_delays"+num2str(M_m)+"_D"+num2str(22)+"_degree3_r"+num2str(r)+".mat", "RMS_prd", "RMS_sim")
        
        results_prd(r,:) = RMS_prd;
        results_sim(r,:) = RMS_sim;
        
        % Increment repeat
        r = r + 1;
%     end    
end
close(dbox) 

results_prd(results_prd == Inf) = NaN;
results_sim(results_sim == Inf) = NaN;

disp("RMS");
[nanmean(results_prd,1); nanmean(results_sim,1)]

disp("Proportion instable");
[mean(isnan(results_prd)); mean(isnan(results_sim))]
