function A_hat = dispResults(causalVars, causalCoeff, fLabel, P)
%DISPRESULTS Displays the inferred causal variables along with their
% causal coefficients in VAR model
% OUTPUT - 
% A_hat := The row corresponding to the target feature in the output
% Adjacency matrix of the feature causal graph

% Threshold of causal coefficients for consideration
th = 0;%1e-10;
A_hat = zeros(1,P);
fprintf('\nFeatures Causally affecting Feature %d :', fLabel(1));
result = [];
for j=1:length(causalVars)
    if abs(causalCoeff(j)) > th
        % fLabel contains the actual odering of features, with the first
        % entry always being the target feature, so fLabel(j) returns the
        % actual index (number) of the j-th causal feature
        result = [result; {fLabel(causalVars(j)), causalCoeff(j)}];
        A_hat(fLabel(causalVars(j))) = 1;
    end
end
fprintf('\nFeatures Absolute_Strength_of_Causality\n');
disp(result);

end

