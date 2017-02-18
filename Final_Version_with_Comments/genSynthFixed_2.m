function CG = genSynthFixed_2()
%GENSYNTHFIXED_2 Generates synthetic time series data with values from Normal
%distribution and then populating the entire series following the VAR model
%which is described below

%% Random Number Generation - Setting
rng('default');

%% VAR Model 1 %%

% A Star graph where all ftrs x_2 to x_P causally affect x_1, but each of
% them has a different time lag effect ~ U(lag_LB, lag_UB)
% # of features (time series) (x_1, x_2, ... , x_P)
% Let x_1 be the target ftr
P = 5;
sig_N = 0.3; % variance of Gaussian Noise process
% VAR equations
% X_i(t) = eta_i(t), for all i = 2 to P
% X_1(t) = a_1*X_1(t - l_1) + a_2*X_2(t - l_2) + a_3*X_3(t - l_3) + ... + a_P*X_P(t - l_p) + eta_1(t) where a_1 = 0
% Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph i.e the indices
% corresponding to entries '1' in A(i,:) are the variables causally
% affecting ftr i
A = zeros(P);
A(1,2:end) = ones(1,P-1);
% lower and upper bounds of time lag for each ftr
lag_LB = 1;
lag_UB = 50;
T = max(2000,5*lag_UB); % length of time series
lags = randi([lag_LB,lag_UB], 1, P); % time lags for ftrs x_1 to x_P
L = max(lags); % L : true maximum time lag
lags = [L, lags(2:end)];
coeff = rand(P,1);%(1/P) + (2/P)*rand(P,1); % coeff : A = [a_1, a_2, ... , a_P]
coeff(1) = 0; % a_1 = 0
% series : a P * T matrix
series = zeros(P,T);
for p = 1:P
    series(p,1:L) = normrnd(0,1,1,L);
end
for t = L+1:T
    for p = 1:P
        if p == 1
            for q = 1:P
                series(p, t) = series(p, t) + coeff(q)*series(q, t-lags(q));
            end
            series(p, t) = series(p, t) + sig_N*randn;
        else
            series(p, t) = randn;
        end
    end
end
winSize = 2*L;
% series : a P * T matrix
series = series(:,winSize+1:end);
T = T-winSize;
% Store the Granger Causal Graph as a struct
CG = struct('adjM', A, 'lags', lags, 'coeff', coeff, 'maxLag', L, 'series', series, 'P', P, 'length', T);
% Save the Causal Graph structure
save('synthData2.mat', 'series', 'CG');


end
