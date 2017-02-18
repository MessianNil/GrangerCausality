function calcF1score(A, B)
%CALCF1SCORE Given the adjacency matrices of the Original (A) and the Output
% (B) Causal Graphs, A, B in P x P, this method calculates
% the Precision, Recall and F1 score
% Notation : Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph 
% Thus the indices corresponding to entries '1' in A(i,:) are the variables
% causally affecting ftr i

%%%% Precision %%%
% (P) = |(i,j) in P x P : A(i,j) = B(i,j) = 1| / |(i,j) in P X P : B(i,j) = 1|
%%%% Recall %%%
% (R) = |(i,j) in P x P : A(i,j) = B(i,j) = 1| / |(i,j) in P X P : A(i,j) = 1|
%%%% F1-score (F-score / F-measure) %%%
% (F) = (2*P*R)/(P+R)

org_indx = find(A); % Find the indices of the entries where A(i,j) = 1
pred_indx = find(B); % Find the indices of the entries where B(i,j) = 1
comm_indx = intersect(org_indx, pred_indx);
if ~isempty(pred_indx) && ~isempty(org_indx)
    prec = length(comm_indx) / length(pred_indx);
    fprintf('Precision (P) : %f \n', prec);
    rcall = length(comm_indx) / length(org_indx);
    fprintf('Recall (R) : %f \n', rcall);
    if (prec+rcall) ~= 0
        f1Score = (2*prec*rcall) / (prec+rcall);
        fprintf('F1-Score (F1) : %f \n', f1Score);
    end
end

end

