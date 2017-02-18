function CG = genSynthFixed_4()
%GENSYNTHFIXED_4 Generates synthetic time series data with values from Normal
%distribution and then populating the entire series following the VAR model
%which is described below

%% Random Number Generation - Setting
rng('default');

%% VAR Model - 1 %%

%%% The 5 variable model as desribed in Sec 3.2 (eq 15) of the Paper -
% Gene regulatory network discovery using pairwise Granger causality
P = 5;
sig_N = 0.3; % variance of Gaussian Noise process
% VAR equations
% x_1(t) = 0.95*sqrt(2)*x_1(t-1) - 0.9025*x_1(t-2) + w_1(t)
% x_2(t) = 0.5*x_1(t-2) + w_2(t)
% x_3(t) = -0.4*x_1(t-3) + w_3(t)
% x_4(t) = -0.5*x_1(t-2) + 0.25*sqrt(2)*x_4(t-1) + 0.25*sqrt(2)*x_5(t-1) + w_4(t)
% x_5(t) = -0.25*sqrt(2)*x_4(t-1) + 0.25*sqrt(2)*x_5(t-1) + w_5(t)
% Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph i.e the indices
% corresponding to entries '1' in A(i,:) are the variables causally
% affecting ftr i
A = [1 0 0 0 0; 1 0 0 0 0; 1 0 0 0 0; 1 0 0 1 1; 0 0 0 1 1];
T = 2000;
lags = [2, 2, 3, 2, 1]; % max lag for each feature 1 to 5
L = max(lags); % L : true maximum time lag in the system
trans_series = zeros(P,T);
for p = 1:P
    trans_series(p,1:L) = normrnd(0,1,1,L);
end
% Time lagged series data for all P variables
for t = L+1:T
    trans_series(1, t) = 0.95*sqrt(2)*trans_series(1, t-1) - 0.9025*trans_series(1, t-2) + sig_N*randn;
    trans_series(2, t) = 0.5*trans_series(1, t-2) + sig_N*randn;
    trans_series(3, t) = -0.4*trans_series(1, t-3) + sig_N*randn;
    trans_series(4, t) = -0.5*trans_series(1, t-2) + 0.25*sqrt(2)*trans_series(4, t-1) + 0.25*sqrt(2)*trans_series(5, t-1) + sig_N*randn;
    trans_series(5, t) = -0.25*sqrt(2)*trans_series(4, t-1) + 0.25*sqrt(2)*trans_series(5, t-1) + sig_N*randn;
end
winSize = 2*L;
% series : a P * T matrix
series = trans_series(:,winSize+1:end);
T = T-winSize;
% Store the Granger Causal Graph as a struct
CG = struct('adjM', A, 'lags', lags, 'maxLag', L, 'series', series, 'P', P, 'length', T);
% Save the Causal Graph structure
save('synthData4a.mat', 'series', 'CG');


%% VAR Model - 2 %%
%{
%%% The 3 variable model as desribed in Sec 3.2 (eq 14) of the Paper -
% Gene regulatory network discovery using pairwise Granger causality
P = 3;
sig_N = 0.3; % variance of Gaussian Noise process
% VAR equations
% x(t) = 0.8*x(t-1) - 0.5*x(t-2) + 0.4*z(t-1) + w_1(t)
% y(t) = 0.9*y(t-1) - 0.8*y(t-2) + w_2(t)
% z(t) = 0.5*z(t-1) - 0.2*z(t-2) + 0.5*y(t-1) + w_3(t)
% Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph i.e the indices
% corresponding to entries '1' in A(i,:) are the variables causally
% affecting ftr i
A = [1 0 1; 0 1 0; 0 1 1];
T = 2000;
lags = [2 2 2]; % max lag for each feature
L = max(lags); % L : true maximum time lag in the system
trans_series = zeros(P,T);
for p = 1:P
    trans_series(p,1:L) = normrnd(0,1,1,L);
end
% Time lagged series data for all P variables
for t = L+1:T
    trans_series(1, t) = 0.8*trans_series(1, t-1) - 0.5*trans_series(1, t-2) + 0.4*trans_series(3, t-1) + sig_N*randn;
    trans_series(2, t) = 0.9*trans_series(2, t-1) - 0.8*trans_series(2, t-2) + sig_N*randn;
    trans_series(3, t) = 0.5*trans_series(3, t-1) - 0.2*trans_series(3, t-2) + 0.5*trans_series(2, t-1) + sig_N*randn;
end
winSize = 2*L;
% series : a P * T matrix
series = trans_series(:,winSize+1:end);
T = T-winSize;
% Store the Granger Causal Graph as a struct
CG = struct('adjM', A, 'lags', lags, 'maxLag', L, 'series', series, 'P', P, 'length', T);
% Save the Causal Graph structure
save('synthData4b.mat', 'series', 'CG');
%}

end
