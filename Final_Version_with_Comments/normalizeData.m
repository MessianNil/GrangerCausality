function X_norm = normalizeData(X)
%NORMALIZEDATA Normalizes the features in X (T * P matrix)
% T = # of samples, P = # of features
% Returns a normalized version of X.
% This is often a good preprocessing step to do when working with learning
%  algorithms

%% Method 1 - Same as zscore in Matlab %%
% The mean of each feature is 0 and the standard deviation is 1. 
% X_norm = X-mu is done here
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

% X_norm = X_norm/sigma is done here
sigma = std(X_norm);

z_ind = find(sigma == 0);
if ~isempty(z_ind)
    X_Z = X_norm(:, z_ind);
    X_norm(:, z_ind) = X_Z;
end
non_z_ind = find(sigma);
if ~isempty(non_z_ind)
    X_nonZ = X_norm(:, non_z_ind);
    sigma = sigma(non_z_ind);
    X_nonZ = bsxfun(@rdivide, X_nonZ, sigma);
    X_norm(:, non_z_ind) = X_nonZ;
end


%% Method 2 %%
%{
P = size(X,2);
X_norm = X;
for p = 1:P
    X_norm(:,p) = (X(:,p)-min(X(:,p)))/(max(X(:,p))-min(X(:,p)));
end
%}

end
