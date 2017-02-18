function CG = genSynthFixed_1()
%GENSYNTHFIXED_1 Generates synthetic time series data with values from Normal
%distribution and then populating the entire series following the VAR model
%which is described below

%% Random Number Generation - Setting
rng('default');

%% VAR Model - 1 %%

P = 4; % P: # of features (time series)
sig_N = 0.3; % variance of Gaussian Noise process
% VAR equations
% X_1(t) = a*X_4(t-2) + eta_1(t)
% X_2(t) = b*X_4(t-1) + c*X_3(t-1) + eta_2(t)
% X_3(t) = eta_3(t)
% X_4(t) = eta_4(t)
% Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph i.e the indices
% corresponding to entries '1' in A(i,:) are the variables causally
% affecting ftr i
A = [0 0 0 1; 0 0 1 1; 0 0 0 0; 0 0 0 0];
T = 2000;
lags = [2, 1, 0, 0]; % max lag for each feature 1 to P
L = max(lags); % L : true maximum time lag in the system
coeff = -1 + 2*rand(3,1); % coeff
trans_series = zeros(P,T);
for p = 1:P
    trans_series(p,1:L) = normrnd(0,1,1,L);
end
% Time lagged series data for all variables
for t = L+1:T
    trans_series(1, t) = coeff(1)*trans_series(4, t-2) + sig_N*randn;
    trans_series(2, t) = coeff(2)*trans_series(4, t-1) + coeff(3)*trans_series(3, t-1) + sig_N*randn;
    trans_series(3, t) = randn;
    trans_series(4, t) = randn;
end
winSize = 2*L;
% series : a P * T matrix
series = trans_series(:,winSize+1:end);
T = T-winSize;
% Store the Granger Causal Graph as a struct
CG = struct('adjM', A, 'lags', lags, 'maxLag', L, 'series', series, 'P', P, 'length', T, 'coeff', coeff);
% Save the Causal Graph structure
save('synthData1a.mat', 'series', 'CG');


%% VAR Model - 2 %%
%{
P = 2; % P: # of features (time series)
sig_N = 0.3; % variance of Gaussian Noise process
% VAR equations
% x(t) = a_1*x(t-1) + a_2*y(t-2) + e_1(t)
% y(t) = a_3*y(t-10) + e_2(t)
% Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph i.e the indices
% corresponding to entries '1' in A(i,:) are the variables causally
% affecting ftr i
A = [1 1; 0 1];
T = 2000;
lags = [2 10]; % max lag for each time series variable
L = max(lags); % L : true maximum time lag in the system
coeff = -1/2 + rand(1,3); % coeff in VAR equations ~ U[-0.5, 0.5]
trans_series = zeros(P,T);
for p = 1:P
    trans_series(p,1:L) = normrnd(0,1,1,L);
end
% Generating time lagged data
for t = L+1:T
    trans_series(1, t) = coeff(1)*trans_series(1, t-1) + coeff(2)*trans_series(1, t-2) + sig_N*randn;
    trans_series(2, t) = coeff(3)*trans_series(2, t-10) + sig_N*randn;
end
winSize = 2*L;
% series : a P * T matrix
series = trans_series(:,winSize+1:end);
T = T-winSize;
% Store the Granger Causal Graph as a struct
CG = struct('adjM', A, 'lags', lags, 'maxLag', L, 'series', series, 'P', P, 'length', T, 'coeff', coeff);
% Save the Causal Graph structure
save('synthData1b.mat', 'series', 'CG');
%}

end

