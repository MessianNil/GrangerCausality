function [Lag, causalVars, causalCoeff] = chooseLag_MSE(index_Series, MSE_V, AIC_V, L_init, epsilon)
%CHOOSELAG_MSE Extracts the causal variables, their coefficients and
% all other information stored in the datastructure index_Series and returns
% the values corresponding to the "best" MSE value

%% Choosing the best Lag value - MSE measure
% Finds the Lag value at which MSE is minimum or find the smallest Lag
% value at which MSE is within additive epsilon bound of the minimum
% MSE (if any)
fprintf('\n%%%%%%%%%%%%%%%%%%%% Using MSE measure %%%%%%%%%%%%%%%%%%%%\n');
[minMSE, minInd] = min(MSE_V);
bestLag = L_init*minInd;
fprintf('\nMinimum MSE is %f', minMSE);
fprintf('\nCorresponding AIC value is %f', AIC_V(minInd));
fprintf('\nLag value corresp. to min. MSE = %d\n', bestLag);
% valIndcs = find(abs(MSE_V-minMSE) <= epsilon*minMSE);
valIndcs = find(abs(MSE_V-minMSE) <= epsilon);
indx = valIndcs(1);
MSE_EPS = MSE_V(indx);
AIC_EPS = AIC_V(indx);
addMSELag = L_init*indx;
fprintf('\nRelative epsilon bound = %f', epsilon);
fprintf('\nChosen MSE (within eps additive) is %f', MSE_EPS);
fprintf('\nCorresponding AIC value is %f', AIC_EPS);
fprintf('\nLag value chosen = %d\n', addMSELag);
% Extracting the Causal Features for a target feature along with the total
% absolute value of their corresponding VAR Causal Coefficients for the
% chosen lag value addMSELag, which in some sense is a measure of strength
% of the causal relationship
causalVars = unique(index_Series{addMSELag/L_init}{1});
causalCoeff = zeros(length(causalVars),1);
totalCausVars = index_Series{addMSELag/L_init}{1};
totalCoeff = index_Series{addMSELag/L_init}{2};
for i =1:length(causalVars)
    for j=1:length(totalCausVars)
        if totalCausVars(j) == causalVars(i)
            causalCoeff(i) = causalCoeff(i) + abs(totalCoeff(j));
            %             causalCoeff(i) = causalCoeff(i) + totalCoeff(j);
        end
    end
end
% Returns the value of Lag (if it exists) at which MSE is within additive
% epsilon bound of the minimum MSE, else returns the lag at which MSE
% is minimum
Lag = addMSELag;
fprintf('\n...............................................\n');




%% Choosing the best Lag value - MSE measure
%{
% Find the Lag value at which MSE is minimum or find the smallest Lag
% value at which MSE is within 1 standard error of the minimum
fprintf('\n%%%%%%%%%%%%%%%%%%%% Using MSE measure %%%%%%%%%%%%%%%%%%%%\n');
[minMSE, minInd] = min(MSE_V);
one_SE = std(MSE_V)/sqrt(numel(MSE_V)-1);
bestLag = L_init*minInd;
fprintf('\nMinimum MSE is %f', minMSE);
fprintf('\nCorresponding AIC value is %f', AIC_V(minInd));
fprintf('\nLag value corresp. to min. MSE = %d\n', bestLag);
valIndcs = find(abs(MSE_V-minMSE) <= one_SE);
indx = valIndcs(1);
MSE_EPS = MSE_V(indx);
AIC_EPS = AIC_V(indx);
addMSELag = L_init*indx;
fprintf('\n1 Standard Error = %f', one_SE);
fprintf('\nChosen MSE (at 1SE) is %f', MSE_EPS);
fprintf('\nCorresponding AIC value is %f', AIC_EPS);
fprintf('\nLag value chosen = %d\n', addMSELag);
% Extracting the Causal Features for a target feature along with the total
% absolute value of their corresponding VAR Causal Coefficients for the
% chosen lag value addMSELag, which in some sense is a measure of strength
% of the causal relationship
causalVars = unique(index_Series{addMSELag/L_init}{1});
causalCoeff = zeros(length(causalVars),1);
totalCausVars = index_Series{addMSELag/L_init}{1};
totalCoeff = index_Series{addMSELag/L_init}{2};
for i =1:length(causalVars)
    for j=1:length(totalCausVars)
        if totalCausVars(j) == causalVars(i)
            causalCoeff(i) = causalCoeff(i) + abs(totalCoeff(j));
            %             causalCoeff(i) = causalCoeff(i) + totalCoeff(j);
        end
    end
end
% Returns the value of Lag (if it exists) at which MSE is within 1 standard
% error bound of the minimum MSE, else returns the lag at which MSE
% is minimum
Lag = addMSELag;
fprintf('\n...............................................\n');
%}

end

