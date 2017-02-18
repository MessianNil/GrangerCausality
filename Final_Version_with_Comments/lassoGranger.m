function [bestCoeff, minMSE, minAIC, minLambda] = lassoGranger(X, y, T, L, lambdas)
%LASSOGRANGER This code finds the Granger causality relationship amongst
% the time series variables solving the Lasso-Granger optimization routine
% OUTPUTS - 
% bestCoeff: coefficient vector
% minMSE: Minimum Mean Squared Error for the chosen value of lambda
% minAIC: Minimum AIC score to select the best value of Lambda
% minLambda : The value of lambda for which AIC is minimum

%% Using AIC metric to chose the best value of lambda
th = 0;
% Choosing the best lambda from several choices of lambda values
if length(lambdas) > 1
    [coeffM, FitInfo] = lasso(X, y, 'Alpha', 1, 'Lambda', lambdas);
    MSEVec = FitInfo.MSE;
    AICVec = zeros(length(lambdas),1);
    % # of observations (data points)
    n = T-L;
    for i=1:length(lambdas)
        % Residual Sum of Squares, rss
        rss = norm(X*coeffM(:,i) - y)^2;
        % k = #degrees of freedom
        k = 1 + sum(abs(coeffM(:,i)) > th);
        % Normalized AIC score := AIC/n
        AICVec(i) = rss/n + (2*k)/n;
        %         AICVec(i) = rss + 2*k;
        %         AICVec(i) = n*log(rss/n) + n*(log(2*pi)+1) + 2*k; %%% When sigma_hat^2 (MLE estimate) is conisdered
        
        %%% AICc - bias correction for finite samples %%%
        if n/k < 40
            AICVec(i) = AICVec(i) + (2*k*(k+1))/(n*(n-k-1));
            %             AICVec(i) = AICVec(i) + (2*k*(k+1))/(n-k-1);
        end
    end
    % Choosing lambda using minimum AIC corresponds to a model selection
    [minAIC, indx] = min(AICVec);
    bestCoeff = coeffM(:,indx);
    minLambda = lambdas(indx);
    minMSE = MSEVec(indx);

else
    %%% OLS-estimate for a fixed value of lambda %%%
    if lambdas == 0
        minLambda = lambdas;
        bestCoeff = pinv(X'*X)*X'*y;
        % k = #degrees of freedom
        k = 1 + sum(abs(bestCoeff) > th);
        % # of observations (data points)
        n = T-L;
        % Residual Sum of Squares, rss
        rss = norm(X*bestCoeff - y)^2;
        % Normalized AIC score := AIC/n
        minAIC = rss/n + (2*k)/n;
        %         minAIC = rss + 2*k;
        %         minAIC = n*log(rss/n) + n*(log(2*pi)+1) + 2*k; %%% When sigma_hat^2 (MLE estimate) is conisdered
        
        %%% AICc - bias correction for finite samples %%%
        if n/k < 40
            minAIC = minAIC + (2*k*(k+1))/(n*(n-k-1));
            %             minAIC = minAIC + (2*k*(k+1))/(n-k-1);
        end
        minMSE = 0.5*rss/(n-k-1);
        
    else
        %%% Lasso-estimate for a fixed value of lambda %%%
        minLambda = lambdas;
        [bestCoeff, FitInfo] = lasso(X, y, 'Alpha', 1, 'Lambda', minLambda);
        % k = #degrees of freedom
        k = 1 + sum(abs(bestCoeff) > th);
        % # of observations (data points)
        n = T-L;
        % Residual Sum of Squares, rss
        rss = norm(X*bestCoeff - y)^2;
        % Normalized AIC score := AIC/n
        minAIC = rss/n + (2*k)/n;
        %         minAIC = rss + 2*k;
        %         minAIC = n*log(rss/n) + n*(log(2*pi)+1) + 2*k; %%% When sigma_hat^2 (MLE estimate) is conisdered
        
        %%% AICc - bias correction for finite samples %%%
        if n/k < 40
            minAIC = minAIC + (2*k*(k+1))/(n*(n-k-1));
            %             minAIC = minAIC + (2*k*(k+1))/(n-k-1);
        end
        minMSE = FitInfo.MSE;
    end
end

end
