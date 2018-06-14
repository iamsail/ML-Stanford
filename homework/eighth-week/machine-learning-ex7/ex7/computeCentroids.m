function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for i = 1 : K  
    all_K = 0;    %用于存储x的和  
    count = sum(idx == i);    %用于存储分配到centroids(i, :)中元素的个数  
    temp_meet = find(idx == i);    %找出分配到centroids(i, :)中所有元素的行索引  
    for j = 1 : numel(temp_meet)     
        all_K = all_K + X(temp_meet(j), :);  
    end  
    centroids(i, :) = all_K / count;  
end  
  
%第二种方法（向量化表示）  
% for i = 1 : K  
%     centroids(i, :) = (X' * (idx == i)) / sum(idx == i);      
%     (idx ==i)目的是将不是i值的X中对应数据变为0.  
% end  






% =============================================================


end

