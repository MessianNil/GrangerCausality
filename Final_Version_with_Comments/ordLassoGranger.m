function [index_Series, MSE_V, AIC_V] = ordLassoGranger(series, L_init, lambdas, MaxLag)
%ORDLASSOGRANGER Computes the Causal Granger lasso with different choices
% of incremental time lags and figures out the best lag value
% This is the Lasso-Granger algorithm with incremental guesses for max lag
% used to model causal relationships amongst time series data without any
% efficient pruning of variables across lag increments

% A normalized (z-scored) T x P matrix, the target feature (time-series) variable is in first column
% T = Length of the dataset (# of observation points)
% P = # of time series variables / features
[T, P] = size(series);
% Initial guess for max lag / model order
currLag = L_init;
numIter = floor(MaxLag/L_init);

% Storing Features causally affecting the input feature along with their causal coefficients in the VAR model
index_Series = cell(numIter, 1);

% Storing the best value of lambda chosen as per minimum AIC / AIC-c score,
% along with the corresponding AIC / AIC-c score and Mean Sqaured Error (MSE)
MSE_V = inf*ones(numIter,1);
AIC_V = inf*ones(numIter,1);
LAMBDA_V = inf*ones(numIter,1);

% Looping with incremental guesses for max lag
while(currLag <= MaxLag)
    % X is a matrix with (T-L) many rows and (P*L) many columns, where L
    % is the current guess for max lag
    X = zeros(T-currLag, P*currLag);
    y = zeros(T-currLag, 1);
    for t = (currLag+1):T
        y(t-currLag) = series(t, 1); % Since the target time-series variable is present in first column
        X(t-currLag, :) = reshape(flipud(series((t-currLag):(t-1), :)), 1, P*currLag); % Forming the (t-L)th row of X
    end
    % Call to Lasso-Granger
    [coeffLasso, mse, aic, lambda] = lassoGranger(X, y, T, currLag, lambdas);
    
    %     fprintf('lambda = %f when lag = %d\n', lambda, currLag);
    
    LAMBDA_V(currLag/L_init) = lambda;
    MSE_V(currLag/L_init) = mse;
    AIC_V(currLag/L_init) = aic;
    
    % Support
    supp = find(coeffLasso);
    % Decoding / Extracting the causal variables from supp since we
    % already know the format in which they are encoded row-wise in X
    ftr_indx = 1 + floor((supp-1)/currLag);
    % Storing the features and their corresponding non-zero coefficients output by Lasso
    index_Series{currLag/L_init} = {ftr_indx, coeffLasso(coeffLasso~=0)};
    % Note max lag guess value is always incremented by the initial guess
    % value - This is the only max lag update rule we follow
    currLag = currLag + L_init;
end

end
