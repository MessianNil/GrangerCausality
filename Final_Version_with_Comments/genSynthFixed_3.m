function CG = genSynthFixed_3(strct)
%GENSYNTHFIXED_3 Generates synthetic time series data with values from Normal
%distribution and then populating the entire series following the VAR model
%which is described below

%% Random Number Generation - Setting
rng('shuffle');

%% VAR Model 1 %%
if strct == 1
    % Co-Parent Structure with P=3 time series variables X, Y, Z where Z is the
    % common parent of X and Y
    P = 3;
    sig_N = 0.3; % variance of Gaussian Noise process
    % VAR equations
    % Z(t) = eta_1(t)
    % X(t) = a*Z(t-lags(1)) + eta_2(t)
    % Y(t) = b*Z(t-lags(2)) + eta_3(t)
    % Adjacency Matrix of the directed Feature Causal graph
    % A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
    % a directed edge from j to i in the feature causal graph i.e the indices
    % corresponding to entries '1' in A(i,:) are the variables causally
    % affecting ftr i
    A = [0 0 1; 0 0 1; 0 0 0];
    % time lags for Z->X and Z->Y
    lag_LB = 1;
    lag_UB = 10;
    % length of time series
    T = max(2000,10*lag_UB);
    lags = randi([lag_LB,lag_UB], P);
    lags = A.*lags;
    % lags = [0 0 0; 0 0 0; 1 2 0];
    L = max(lags(:)); % L : true maximum time lag
    coeff = rand(P).*lags; % coeff ~ U(0,1)
    % series_1 : a P * T matrix
    series_1 = zeros(P,T);
    for p = 1:P
        series_1(p,1:L) = normrnd(0,1,1,L);
    end
    for t = L+1:T
        for p = 1:P
            for q = 1:P
                if A(p,q) ~= 0
                    series_1(p, t) = series_1(p, t) + coeff(p,q)*series_1(q, t-lags(p,q));
                end
            end
            series_1(p, t) = series_1(p, t) + sig_N*randn;
        end
    end
    winSize = 2*L;
    % series_1 : a P * T matrix
    series_1 = series_1(:,winSize+1:end);
    T = T-winSize;
    % Store the Granger Causal Graph as a struct
    CG_1 = struct('adjM', A, 'lags', lags, 'coeff', coeff, 'maxLag', L, 'series', series_1, 'P', P, 'length', T);
    CG = CG_1;
    % Save the Causal Graph structure
    save('synthData3_a.mat', 'series_1', 'CG_1');
    
%% VAR Model 2 %%
elseif strct == 2
    % Collider Structure with P=3 time series variables X, Y, Z where Z is the
    % common child of X and Y
    P = 3;
    sig_N = 0.3; % variance of Gaussian Noise process
    % VAR equations
    % X(t) = eta_1(t)
    % Y(t) = eta_2(t)
    % Z(t) = a*X(t-lags(1)) + b*Y(t-lags(2)) + eta_3(t)
    % Adjacency Matrix of the directed Feature Causal graph
    % A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
    % a directed edge from j to i in the feature causal graph i.e the indices
    % corresponding to entries '1' in A(i,:) are the variables causally
    % affecting ftr i
    A = [0 0 0; 0 0 0; 1 1 0];
    % time lags for X->Z and Y->Z
    lag_LB = 1;
    lag_UB = 10;
    % length of time series
    T = max(2000,10*lag_UB);
    lags = randi([lag_LB,lag_UB], P);
    lags = A.*lags;
    % lags = [0 0 0; 0 0 0; 1 2 0];
    L = max(lags(:)); % L : true maximum time lag
    coeff = rand(P).*lags; % coeff ~ U(0,1)
    % series_2 : a P * T matrix
    series_2 = zeros(P,T);
    for p = 1:P
        series_2(p,1:L) = normrnd(0,1,1,L);
    end
    for t = L+1:T
        for p = 1:P
            for q = 1:P
                if A(p,q) ~= 0
                    series_2(p, t) = series_2(p, t) + coeff(p,q)*series_2(q, t-lags(p,q));
                end
            end
            series_2(p, t) = series_2(p, t) + sig_N*randn;
        end
    end
    winSize = 2*L;
    % series_2 : a P * T matrix
    series_2 = series_2(:,winSize+1:end);
    T = T-winSize;
    % Store the Granger Causal Graph as a struct
    CG_2 = struct('adjM', A, 'lags', lags, 'coeff', coeff, 'maxLag', L, 'series', series_2, 'P', P, 'length', T);
    CG = CG_2;
    % Save the Causal Graph structure
    save('synthData3_b.mat', 'series_2', 'CG_2');
    
%% VAR Model 3 %%
else
    % Chain Structure with P=3 time series variables X, Y, Z where X is the
    % parent of Z which in turn is the parent of Y
    P = 3;
    sig_N = 0.3; % variance of Gaussian Noise process
    % VAR equations
    % X(t) = eta_1(t)
    % Y(t) = b*Z(t-lags(2)) + eta_3(t)
    % Z(t) = a*X(t-lags(1)) + eta_2(t)
    % Adjacency Matrix of the directed Feature Causal graph
    % A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
    % a directed edge from j to i in the feature causal graph i.e the indices
    % corresponding to entries '1' in A(i,:) are the variables causally
    % affecting ftr i
    A = [0 0 0; 0 0 1; 1 0 0];
    % time lags for X->Z and Z->Y
    lag_LB = 1;
    lag_UB = 10;
    % length of time series
    T = max(2000,10*lag_UB);
    lags = randi([lag_LB,lag_UB], P);
    lags = A.*lags;
    % lags = [0 0 0; 0 0 0; 1 2 0];
    L = max(lags(:)); % L : true maximum time lag
    coeff = rand(P).*lags; % coeff ~ U(0,1)
    % series_3 : a P * T matrix
    series_3 = zeros(P,T);
    for p = 1:P
        series_3(p,1:L) = normrnd(0,1,1,L);
    end
    for t = L+1:T
        for p = 1:P
            for q = 1:P
                if A(p,q) ~= 0
                    series_3(p, t) = series_3(p, t) + coeff(p,q)*series_3(q, t-lags(p,q));
                end
            end
            series_3(p, t) = series_3(p, t) + sig_N*randn;
        end
    end
    winSize = 2*L;
    % series_3 : a P * T matrix
    series_3 = series_3(:,winSize+1:end);
    T = T-winSize;
    % Store the Granger Causal Graph as a struct
    CG_3 = struct('adjM', A, 'lags', lags, 'coeff', coeff, 'maxLag', L, 'series', series_3, 'P', P, 'length', T);
    CG = CG_3;
    % Save the Causal Graph structure
    save('synthData3_c.mat', 'series_3', 'CG_3');
end

end
