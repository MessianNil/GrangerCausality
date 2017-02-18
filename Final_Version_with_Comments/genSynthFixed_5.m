function CG = genSynthFixed_5()
%GENSYNTHFIXED_5 Generates synthetic time series data with values from Normal
%distribution and then populating the entire series following the VAR model
%which is described below

%% Random Number Generation - Setting
rng('default');

%% VAR Model 1 %%

P = 2;
sig_N = 0.3; % variance of Gaussian Noise process
% VAR equations
% x(t) = a_1*x(t-1) - a_2*x(t-2) + e_1(t)
% y(t) = a_3*y(t-4) + e_2(t)
% Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph i.e the indices
% corresponding to entries '1' in A(i,:) are the variables causally
% affecting ftr i
A = [1 0; 0 1];
T = 2000;
lags = [2 4]; % max lag for each time series variable
L = max(lags); % L : true maximum time lag in the system
coeff = -1 + 2*rand(1,3);%[1 1 1];
trans_series = zeros(P,T);
for p = 1:P
    trans_series(p,1:L) = normrnd(0,1,1,L);
end
% Generating time lagged data
for t = L+1:T
    trans_series(1, t) = coeff(1)*trans_series(1, t-1) - coeff(2)*trans_series(1, t-2) + sig_N*randn;
    trans_series(2, t) = coeff(3)*trans_series(2, t-4) + sig_N*randn;
end
winSize = 2*L;
% series : a P * T matrix
series = trans_series(:,winSize+1:end);
T = T-winSize;
% Store the Granger Causal Graph as a struct
CG = struct('adjM', A, 'lags', lags, 'maxLag', L, 'series', series, 'P', P, 'length', T, 'coeff', coeff);
% Save the Causal Graph structure
save('synthData5a.mat', 'series', 'CG');
%}

%% VAR Model 2 %%
%{
P = 2;
sig_N = 0.3; % variance of Gaussian Noise process
% VAR equations
% x(t) = a_1*y(t-1) + e_1(t)
% y(t) = a_2*y(t-2) + e_2(t)
% Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph i.e the indices
% corresponding to entries '1' in A(i,:) are the variables causally
% affecting ftr i
A = [0 1; 0 1];
T = 2000;
lags = [1 2]; % max lag for both predictors x and y
L = max(lags); % L : true maximum time lag in the system
coeff = rand(1,2);%[1 1];
trans_series = zeros(P,T);
for p = 1:P
    trans_series(p,1:L) = normrnd(0,1,1,L);
end
% Time lagged data
for t = L+1:T
    trans_series(1, t) = coeff(1)*trans_series(2, t-1) + sig_N*randn;
    trans_series(2, t) = coeff(2)*trans_series(2, t-2) + sig_N*randn;
end
winSize = 2*L;
% series : a P * T matrix
series = trans_series(:,winSize+1:end);
T = T-winSize;
% Store the Granger Causal Graph as a struct
CG = struct('adjM', A, 'lags', lags, 'maxLag', L, 'series', series, 'P', P, 'length', T, 'coeff', coeff);
% Save the Causal Graph structure
save('synthData5b.mat', 'series', 'CG');
%}

%% VAR Model 3 %%
%{
%%% The 2 variable model as desribed in Sec 3.2 (Fig. 2) of the Paper -
% Gene regulatory network discovery using pairwise Granger causality
P = 2;
sig_N = 0.3; % variance of Gaussian Noise process
% VAR equations
% y(t) = a_1*y(t-1) + a_2*y(t-2) + e_1(t)
% z(t) = a_3*z(t-1) + a_4*z(t-2) + a_5*y(t-1) + e_2(t)
% Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph i.e the indices
% corresponding to entries '1' in A(i,:) are the variables causally
% affecting ftr i
A = [1 0; 1 1];
T = 2000;
lags = [2, 2]; % max lag for both predictors y and z
L = max(lags); % L : true maximum time lag in the system
coeff = -1/3 + (2/3)*rand(1,5);%1 - 2*rand(1,5);%[1/4 1/4 1/6 1/6 1/6];
trans_series = zeros(P,T);
for p = 1:P
    trans_series(p,1:L) = normrnd(0,1,1,L);
end
% Time lagged data
for t = L+1:T
    trans_series(1, t) = coeff(1)*trans_series(1, t-1) + coeff(2)*trans_series(1, t-2) + sig_N*randn;
    trans_series(2, t) = coeff(3)*trans_series(2, t-1) + coeff(4)*trans_series(2, t-2) + coeff(5)*trans_series(1, t-1) + sig_N*randn;
end
winSize = 2*L;
% series : a P * T matrix
series = trans_series(:,winSize+1:end);
T = T-winSize;
% Store the Granger Causal Graph as a struct
CG = struct('adjM', A, 'lags', lags, 'maxLag', L, 'series', series, 'P', P, 'length', T, 'coeff', coeff);
% Save the Causal Graph structure
save('synthData5c.mat', 'series', 'CG');
%}

end
