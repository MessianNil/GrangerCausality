function [index_Series, MSE_V, AIC_V] = groupCausalGranger(series, L_init, lambdas, MaxLag)
%GROUPCAUSALGRANGER Computes the Causal Granger Group lasso with different choices
% of incremental time lags and figures out the best lag value using aic metric
% This is the Group-Lasso-Granger++ algorithm for modeling causal relationships
% amongst time series data

% A normalized (z-scored) T x P matrix, the target feature (time-series) variable is in first column
% T = Length of the dataset (# of observation points)
% P = # of time series variables / features
[T, P] = size(series);
% Initial guess for max lag / model order
currLag = L_init;
numIter = floor(MaxLag/L_init);

% Storing Features causally affecting the input feature along with their causal coefficients in the VAR model
index_Series = cell(numIter, 1);
% Retains the supporting columns (corresp. to non-zero Group-Lasso coeff.) from previous iteration of currLag
nonZeroCols = [];
rPastWinSize = 0;

MSE_V = inf*ones(numIter,1);
AIC_V = inf*ones(numIter,1);
LAMBDA_V = inf*ones(numIter,1);

% Maintains the group structure in the entire time series with P number of
% groups
Group = cell(1,P);

% Looping with incremental guesses for max lag
while(currLag <= MaxLag)
    % X is a matrix with (T-L) many rows and (P*L_init + |non-zero cols.
    % from previous iteration of maxlag|) many columns, where L is the
    % current guess for max lag and L_init was the initial guess value for
    % max lag
    X = zeros(T-currLag, rPastWinSize+(L_init*P));
    y = zeros(T-currLag, 1);
    if currLag == L_init
        for t = (currLag+1):T
            y(t-currLag) = series(t, 1); % Since the target time-series variable is present in first column
            X(t-currLag, :) = reshape(flipud(series((t-L_init):(t-1), :)), 1, P*L_init); % Forming the (t-L)th row of X
        end
        % Group{g} is a cell array containing the column indices of
        % features belonging to that group
        for g = 1:P
            Group{g} = (g-1)*L_init+1:g*L_init;
        end
    else % currLag > L_init
        for t = (currLag+1):T
            y(t-currLag) = series(t, 1);
            % Taking all the time lagged variables from old past (i.e t-CL to t-PL-1)
            % and only non-zero variables from recent past (i.e t-PL to t-1)
            % which have already been found with previous iteration of max lag
            % CL = current guess for max lag, PL = previous guess for max lag
            prevLag = currLag-L_init;
            rPSparse = nonZeroCols(t-prevLag, :);
            oPAll = reshape(flipud(series((t-currLag):(t-prevLag-1), :)), 1, P*L_init);
            X(t-currLag, :) = [rPSparse, oPAll];
        end
    end
    % Call to Group-Lasso-Granger
    Grp = getGroupIndexed(Group);
    [coeffGroupLasso, mse, aic, lambda] = groupLassoGranger(X, y, T, currLag, lambdas, Grp);
    
    %     fprintf('lambda = %f when lag = %d\n', lambda, currLag);
    
    LAMBDA_V(currLag/L_init) = lambda;
    MSE_V(currLag/L_init) = mse;
    AIC_V(currLag/L_init) = aic;
    
    % Retains the supporting columns (corresp. to non-zero Group-Lasso coeff.) from previous iteration of currLag
    nonZeroCols = X(:, (coeffGroupLasso ~= 0));
    % Support of the recent past. Note with L_init, supp_rPS is empty since rPastWinSize = 0
    supp_rPS = find(coeffGroupLasso(1:rPastWinSize));
    % Support of the old past
    supp_oPAll = find(coeffGroupLasso((rPastWinSize+1):end));
    % Decoding / Extracting the causal variables from supp_oPAll since we
    % already know the format in which they are encoded row-wise in X
    ftr_indx_oPAll = 1 + floor((supp_oPAll-1)/L_init);
    if ~isempty(supp_rPS)
        ftr_indx_rPS = index_Series{currLag/L_init - 1}{1};
        ftr_index_series = [ftr_indx_rPS(supp_rPS); ftr_indx_oPAll];
    else
        ftr_index_series = ftr_indx_oPAll;
    end
    % Storing the features and their corresponding non-zero coefficients output by Group Lasso
    index_Series{currLag/L_init} = {ftr_index_series, coeffGroupLasso(coeffGroupLasso~=0)};
    % Updating the rPastWinSize to the number of non-zero cols. from
    % the Lasso estimate with the current guess value for max lag
    rPastWinSize = size(nonZeroCols, 2);
    
    % Forming Groups for the next guess of maxlag
    % Remember Group{g} is a cell array containing the column indices of 
    % features belonging to that group
    for g = 1:P
        rPSColIndx = [];
        for rCi = 1:length(ftr_index_series)
            if ftr_index_series(rCi) == g
                rPSColIndx = [rPSColIndx, rCi];
            end
        end
        if ~isempty(rPSColIndx)
            Group{g} = [rPSColIndx, (rPastWinSize+(g-1)*L_init+1):(rPastWinSize+g*L_init)];
        else
            Group{g} = (rPastWinSize+(g-1)*L_init+1):(rPastWinSize+g*L_init);
        end
    end
    
    % Note max lag guess value is always incremented by the initial guess
    % value - This is the only max lag update rule we follow
    currLag = currLag + L_init;
end

end

function G = getGroupIndexed(Group)
% Returns a row vector G of size k ( k : the effective number of features of X for a
% given lag) where there are P groups (P : the original number of features /
% time series)
% Columns belonging to a group 'i' are indexed as 'i' in their respective
% column positions in X.
nCols = max(cell2mat(Group));
G = zeros(1,nCols);
% Number of Groups = P
P = length(Group);
for i = 1:P
    tmp = Group{i};
    for j = 1:length(tmp)
        G(tmp(j)) = i;
    end
end

end
