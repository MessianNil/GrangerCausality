clear; close all; clc;
warning off;

%% Generates Synthetic time series data
CG = genSynthFixed_2();
adjM = CG.adjM;
% series : a T * P matrix
series = CG.series';
P = CG.P;
T = CG.length;
L = CG.lags;
fprintf('\nSynthetic Dataset generated ...');
% % series : a normalized (T * P) matrix
normS = normalizeData(series);
series = normS;

%% Lasso-Granger++ Causality for all the features

diary('SynthLog_2.txt');
diary on;
fprintf('\n...................... Lasso-Granger++ ......................');
adjM_hat_LG = zeros(P);
causal_Info_LG = cell(P,1);

% Parameters of Lasso-Granger++ Model
L_init = 1; % Initial guess for time Lag
% lambdas = 0;
lambdas = [0.0001:0.0001:0.0009, 0.001:0.001:0.009, 0.01:0.01:0.09, 0.1:0.1:0.9, 1:1:20];
MaxLag = 25+max(L); % Maximum guess for time Lag

for ftr = 1:1%P
    tic;
    fprintf('\n............. Causal analysis for Time Series # %d .............\n', ftr);
    indx = [ftr, 1:(ftr-1), (ftr+1):P];
    [index_Series_LG, MSE_V, AIC_V] = causalGranger(series(:,indx), L_init, lambdas, MaxLag);
    [Lag_AIC, causalVars_AIC, causalCoeff_AIC] = chooseLag_AIC(index_Series_LG, MSE_V, AIC_V, L_init, 0.05);
    [Lag_MSE, causalVars_MSE, causalCoeff_MSE] = chooseLag_MSE(index_Series_LG, MSE_V, AIC_V, L_init, 0.05);
    dispTrueCause(L, adjM, ftr);
    adjM_hat_LG(ftr,:) = dispResults(causalVars_AIC, causalCoeff_AIC, indx, P);
    causal_Info_LG{ftr} = index_Series_LG;
    
    %%% Plots %%%
    figure;
    subplot(1,2,1);
    plot(L_init:L_init:MaxLag, AIC_V, 'rs-');
    hold on;
    plot(Lag_AIC, AIC_V(Lag_AIC/L_init), 'b*', 'MarkerSize', 12);
    xlabel('Different guesses for Maximum Time Lag (L)');
    ylabel('Performance metric (AIC)');
    title(sprintf('Lasso-Granger++ Error curve for feature %d', ftr));
    legend('Normalized AIC', 'Chosen Lag value');
    hold off;
    
    %     figure;
    subplot(1,2,2);
    plot(L_init:L_init:MaxLag, MSE_V, 'bs-');
    hold on;
    plot(Lag_MSE, MSE_V(Lag_MSE/L_init), 'r*', 'MarkerSize', 12);
    xlabel('Different guesses for Maximum Time Lag (L)');
    ylabel('Performance metric (MSE)');
    title(sprintf('Lasso-Granger++ Error curve for feature %d', ftr));
    legend('Mean Sqaured Error', 'Chosen Lag value');
    hold off;
    
    toc;
end
fprintf('\n......................................\n');
calcF1score(adjM, adjM_hat_LG);
fprintf('......................................\n');
diary off;


%% Group-Lasso-Granger++ Causality for all the features

diary('SynthLog_2.txt');
diary on;
fprintf('\n...................... Group Lasso-Granger++ ......................');
adjM_hat_GLG = zeros(P);
causal_Info_GLG = cell(P,1);

% Parameters of Group-Lasso-Granger++ Model
L_init = 1; % Initial guess for time Lag
% lambdas = 0;
lambdas = [0.0001:0.0001:0.0009, 0.001:0.001:0.009, 0.01:0.01:0.09, 0.1:0.1:0.9, 1:1:20];
MaxLag = 25+max(L); % Maximum guess for time Lag

for ftr = 1:1%P
    tic;
    
    fprintf('\n............. Causal analysis for Time Series # %d .............\n', ftr);
    indx = [ftr, 1:(ftr-1), (ftr+1):P];
    [index_Series_GLG, MSE_V, AIC_V] = groupCausalGranger(series(:,indx), L_init, lambdas, MaxLag);
    [Lag_AIC, causalVars_AIC, causalCoeff_AIC] = chooseLag_AIC(index_Series_GLG, MSE_V, AIC_V, L_init, 0.01);
    [Lag_MSE, causalVars_MSE, causalCoeff_MSE] = chooseLag_MSE(index_Series_GLG, MSE_V, AIC_V, L_init, 0.01);
    dispTrueCause(L, adjM, ftr);
    adjM_hat_GLG(ftr,:) = dispResults(causalVars_AIC, causalCoeff_AIC, indx, P);
    causal_Info_GLG{ftr} = index_Series_GLG;
    
    %%% Plots %%%
    figure;
    subplot(1,2,1);
    plot(L_init:L_init:MaxLag, AIC_V, 'rs-');
    hold on;
    plot(Lag_AIC, AIC_V(Lag_AIC/L_init), 'b*', 'MarkerSize', 12);
    xlabel('Different guesses for Maximum Time Lag (L)');
    ylabel('Performance metric (AIC)');
    title(sprintf('Group-Lasso-Granger++ Error curve for feature %d', ftr));
    legend('Normalized AIC', 'Chosen Lag value');
    hold off;

%     figure;
    subplot(1,2,2);
    plot(L_init:L_init:MaxLag, MSE_V, 'bs-');
    hold on;
    plot(Lag_MSE, MSE_V(Lag_MSE/L_init), 'r*', 'MarkerSize', 12);
    xlabel('Different guesses for Maximum Time Lag (L)');
    ylabel('Performance metric (MSE)');
    title(sprintf('Group-Lasso-Granger++ Error curve for feature %d', ftr));
    legend('Mean Sqaured Error', 'Chosen Lag value');
    hold off;
    
    toc;
end
fprintf('\n......................................\n');
calcF1score(adjM, adjM_hat_GLG);
fprintf('......................................\n');
diary off;


%% Restoring Unnormalized series - a T * P matrix
series = CG.series';

% Clear temporary variables
clearvars ftr indx MSE_V AIC_V;
clearvars causalVars_MSE causalCoeff_MSE causalVars_AIC causalCoeff_AIC;
clearvars index_Series_LG index_Series_GLG;
