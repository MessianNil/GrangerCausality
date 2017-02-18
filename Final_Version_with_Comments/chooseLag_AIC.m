function [Lag, causalVars, causalCoeff] = chooseLag_AIC(index_Series, MSE_V, AIC_V, L_init, epsilon)
%CHOOSELAG_AIC Extracts the causal variables, their coefficients and
% all other information stored in the datastructure index_Series and returns
% the values corresponding to the "best" AIC value

%% Choosing the best Lag value - AIC measure
% Finds the Lag value at which AIC is minimum or find the smallest Lag
% value at which AIC is within additive epsilon bound of the minimum
% AIC (if any)
fprintf('\n%%%%%%%%%%%%%%%%%%%% Using AIC measure %%%%%%%%%%%%%%%%%%%%\n');
[minAIC, minInd] = min(AIC_V);
Lag = L_init*minInd;
fprintf('\nMinimum AIC value is %f', minAIC);
fprintf('\nCorresponding MSE value is %f', MSE_V(minInd));
fprintf('\nLag value corresp. to min. AIC = %d\n', Lag);
valIndcs = find(abs(AIC_V-minAIC) <= epsilon*minAIC);
% valIndcs = find(abs(AIC_V-minAIC) <= epsilon);
indx = valIndcs(1);
MSE_EPS = MSE_V(indx);
AIC_EPS = AIC_V(indx);
addAICLag = L_init*indx;
fprintf('\nEpsilon bound = %f', epsilon);
fprintf('\nChosen AIC value is %f', AIC_EPS);
fprintf('\nCorresponding MSE is %f', MSE_EPS);
fprintf('\nLag value chosen = %d\n', addAICLag);
% Extracting the Causal Features for a target feature along with the total
% absolute value of their corresponding VAR Causal Coefficients for the
% chosen lag value addAICLag, which in some sense is a measure of strength
% of the causal relationship
causalVars = unique(index_Series{addAICLag/L_init}{1});
causalCoeff = zeros(length(causalVars),1);
totalCausVars = index_Series{addAICLag/L_init}{1};
totalCoeff = index_Series{addAICLag/L_init}{2};
for i =1:length(causalVars)
    for j=1:length(totalCausVars)
        if totalCausVars(j) == causalVars(i)
            causalCoeff(i) = causalCoeff(i) + abs(totalCoeff(j));
            %             causalCoeff(i) = causalCoeff(i) + totalCoeff(j);
        end
    end
end
% Returns the value of Lag (if it exists) at which AIC is within additive
% epsilon bound of the minimum AIC score, else returns the lag at which AIC
% is minimum
Lag = addAICLag;
fprintf('\n...............................................\n');





%% Choosing the best Lag value - AIC measure
%{
% Find the Lag value at which AIC is minimum or find the smallest Lag
% value at which AIC is within 1 standard error of the minimum
fprintf('\n%%%%%%%%%%%%%%%%%%%% Using AIC measure %%%%%%%%%%%%%%%%%%%%\n');
[minAIC, minInd] = min(AIC_V);
one_SE = std(AIC_V)/sqrt(numel(AIC_V)-1);
Lag = L_init*minInd;
fprintf('\nMinimum AIC value is %f', minAIC);
fprintf('\nCorresponding MSE value is %f', MSE_V(minInd));
fprintf('\nLag value corresp. to min. AIC = %d\n', Lag);
valIndcs = find(abs(AIC_V-minAIC) <= one_SE);
indx = valIndcs(1);
MSE_EPS = MSE_V(indx);
AIC_EPS = AIC_V(indx);
addAICLag = L_init*indx;
fprintf('\n1 Standard Error = %f', one_SE);
fprintf('\nChosen MSE (at 1 SE) is %f', MSE_EPS);
fprintf('\nCorresponding AIC value is %f', AIC_EPS);
fprintf('\nLag value chosen = %d\n', addAICLag);
% Extracting the Causal Features for a target feature along with the total
% absolute value of their corresponding VAR Causal Coefficients for the
% chosen lag value addAICLag, which in some sense is a measure of strength
% of the causal relationship
causalVars = unique(index_Series{addAICLag/L_init}{1});
causalCoeff = zeros(length(causalVars),1);
totalCausVars = index_Series{addAICLag/L_init}{1};
totalCoeff = index_Series{addAICLag/L_init}{2};
for i =1:length(causalVars)
    for j=1:length(totalCausVars)
        if totalCausVars(j) == causalVars(i)
            causalCoeff(i) = causalCoeff(i) + abs(totalCoeff(j));
            %             causalCoeff(i) = causalCoeff(i) + totalCoeff(j);
        end
    end
end
% Returns the value of Lag (if it exists) at which AIC is within 1 standard
% error bound of the minimum AIC score, else returns the lag at which AIC
% is minimum
Lag = addAICLag;
fprintf('\n...............................................\n');
%}

end

