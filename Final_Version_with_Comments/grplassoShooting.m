% Group Lasso with shooting algorithm
% Author: Xiaohui Chen (xhchen@illinois.edu)
% Department of Statistics
% University of Illinois at Urbana-Champaign
% Version: 2012-Feb

% 3 modifications have been made to this funtion - Modifications 1 and 2 have
% been mentioned at appropriate places in this code, and the 3rd
% modification being the return values - instead of just retuning b, we are
% now also returning the rss

function [b, rss] = grplassoShooting(X, Y, G, lambda, maxIt, tol, standardize)

% Shooting algorithm for the group lasso in the penalized form.
% minimize .5*||Y-X*beta||_2^2 + lambda*sum(sqrt(p_g)*||beta_g||_2)
% where p_g is the dimension of the subspace g, for g in G.
% Ref:
% - Yuan and Lin (2005) Model selection and estimation in regression
% with grouped variables. JRSSB.
% - Fu (1998) Penalized regression: the bridge versus the lasso. J. Comput.
% Graph. Stats.

if nargin < 7, standardize = true; end
if nargin < 6, tol = 1e-10; end
if nargin < 5, maxIt = 1e4; end

% Number of groups
nG = length(unique(G));
[n,p] = size(X);
for g = 1:nG,
    if standardize,
        %         X(:,g==G) = orth(X(:,g==G));    % Caution: orth may flip signs
        
        %%% Modification 1 -- Using Modified Gram-Schmidt Orthonomalization %%%
        X(:,g==G) = GSOrth(X(:,g==G));
    end
	% Dimension for each subspace, assuming full column ranks
    p_g(g) = sum(g==G);
    %{
    %%% Modification 2 -- Weight of each group is 1 for our purpose %%%
    p_g(g) = 1;
	%}
end
if standardize, Y = Y-mean(Y); end

% Initialization
if p > n,
    % From the null model, if p > n
    b = zeros(p,1);
else
    % From the OLS estimate, if p <= n
    b = X \ Y;
end
b_old = b;
i = 0;

% Precompute X'X and X'Y
XTX = X'*X;
XTY = X'*Y;

% Shooting loop
while i < maxIt,
    i = i+1;
    for g = 1:nG,
        S0 = XTY(g==G) - XTX(g==G,g~=G)*b(g~=G);
        if (lambda*sqrt(p_g(g))) < norm(S0,2),
            b(g==G) = (1-lambda*sqrt(p_g(g))/norm(S0,2))*S0;
        else
            b(g==G) = zeros(p_g(g),1);
        end
    end
    % Norm change during successive iterations
    delta = norm(b-b_old,2);
    if delta < tol, break; end
    b_old = b;
end
if i == maxIt,
    fprintf('%s\n', 'Maximum number of iteration reached, shooting may not converge.');
end

%%% Modification 3 -- Returning the residual sum of Squares error corresponding
% to the group-wise orthonormalized X and centered Y %%%
% Returning the Residual Sum of Squares along with the coefficient vector
rss = norm(Y - X*b)^2;

end

%{
% Normalize columns of X to have mean zero and length one.
function sX = normalize(X)

[n,p] = size(X);
sX = X-repmat(mean(X),n,1);
sX = sX*diag(1./sqrt(ones(1,n)*sX.^2));

end
%}
