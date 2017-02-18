function [bestCoeff, minMSE, minAIC, minLambda] = groupLassoGranger(X, y, T, L, lambdas, Group)
%GROUPLASSOGRANGER This code finds the Group Lasso Granger causality relationship
% among the input time series solving the Group-Lasso-Granger optimization routine
% OUTPUTS -
% bestCoeff: coefficient vector
% minMSE: Minimum Mean Squared Error for the chosen value of lambda
% minAIC: Minimum AIC score to select the best value of Lambda
% minLambda : The value of lambda for which AIC is minimum

%% Using AIC metric to chose the best value of lambda
th = 0;
% Choosing the best lambda from several choices of lambda values
if length(lambdas) > 1
    nLambda = length(lambdas);
    nCoeff = size(X,2);
    % coeffM : A Matrix where there are as many columns as the different
    % choices of lambda values, the i-th column stores the coefficient
    % output by the Group Lasso routine with the i-th choice of lambda
    coeffM = zeros(nCoeff,nLambda);
    MSEVec = zeros(nLambda,1);
    AICVec = zeros(nLambda,1);
    % # of observations (data points)
    n = T-L;
    for indx = 1:nLambda
        % Calling the Group-Lasso-Shooting algorihtm implemented by Xiaohui Chen
        % Department of Statistics, University of Illinois at Urbana-Champaign
        % The code has been slightly modified to return the Residual Sum of Squares, rss
        % along with the coefficient vector
        [coeffM(:,indx), rss] = grplassoShooting(X, y, Group, lambdas(indx));
        % k = #degrees of freedom
        k = 1 + sum(abs(coeffM(:,indx)) > th);
        % Normalized AIC score := AIC/n
        AICVec(indx) = rss/n + (2*k)/n;
        %         AICVec(indx) = rss + 2*k;
        %         AICVec(indx) = n*log(rss/n) + n*(log(2*pi)+1) + 2*k; %%% When sigma_hat^2 (MLE estimate) is conisdered
        
        %%% AICc - bias correction for finite samples %%%
        if n/k < 40
            AICVec(indx) = AICVec(indx) + (2*k*(k+1))/(n*(n-k-1));
            %             AICVec(indx) = AICVec(indx) + (2*k*(k+1))/(n-k-1);
        end
        MSEVec(indx) = 0.5*rss/(n-k-1);
        %         MSEVec(indx) = 0.5*rss/n;
    end
    % Choosing lambda using minimum AIC corresponds to a model selection
    [minAIC, minIndx] = min(AICVec);
    bestCoeff = coeffM(:,minIndx);
    minLambda = lambdas(minIndx);
    minMSE = MSEVec(minIndx);
    
else
    %%% For a fixed value of lambda %%%
    minLambda = lambdas;
    % Calling the Group-Lasso-Shooting algorihtm implemented by Xiaohui Chen
    % Department of Statistics, University of Illinois at Urbana-Champaign
    % The code has been slightly modified to return the Residual Sum of Squares, rss
    % along with the coefficient vector
    [bestCoeff, rss] = grplassoShooting(X, y, Group, minLambda);
    % k = #degrees of freedom
    k = 1 + sum(abs(bestCoeff) > th);
    % # of observations (data points)
    n = T-L;
    % Normalized AIC score := AIC/n
    minAIC = rss/n + (2*k)/n;
    %     minAIC = rss + 2*k;
    %     minAIC = n*log(rss/n) + n*(log(2*pi)+1) + 2*k; %%% When sigma_hat^2 (MLE estimate) is conisdered
    
    %%% AICc - bias correction for finite samples %%%
    if n/k < 40
        minAIC = minAIC + (2*k*(k+1))/(n*(n-k-1));
        %         minAIC = minAIC + (2*k*(k+1))/(n-k-1);
    end
    minMSE = 0.5*rss/(n-k-1);
    %         minMSE = 0.5*rss/n;
end

end

