function dispTrueCause(L, A, ftr)
%DISPTRUECAUSE Displays the true causal information
% Adjacency Matrix of the directed Feature Causal graph
% A(i,j) = 1 implies ftr i is causally affected by feature j, i.e there is
% a directed edge from j to i in the feature causal graph i.e the indices
% corresponding to entries '1' in A(i,:) are the variables causally
% affecting ftr i
% Note : Our Algorithm always reports the maximum lag value for each feature

fprintf('\nTrue max lag for feature %d :', ftr);
if isvector(L)
    disp(L(ftr));
else
    disp(max(L(ftr,:)));
end
fprintf('\nTrue Causal info for feature %d :', ftr);
fprintf('\nFeatures causally affecting ftr # %d : \n', ftr);
disp(find(A(ftr,:)));

end

