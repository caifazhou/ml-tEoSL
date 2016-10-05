function [outputArgs] = KNNEstimated(testData, trainX, trainY, nearestNeighbors, distanceMeasurement)
% To determin the class using k nearest neighbors
% [outputArgs] = LSDecision(testData, LSPara, decisionCondition)
% each column is a data point;
% input variables:
%   testData: 
%   trainX:
%   trainY:
%   nearestNeigbors:
%   distanceMeasurement:


% output variables (as a struct):
%   KNN parameters: k
%   estimatedClass:

% Author: CZ
% Version: 1.0
% Date              Status
% 26.09.2016        Draft

    if nargin < 3 || nargin > 5
        error('The number of inputs does not match');
    end;
    
    if nargin == 3 || isempty(distanceMeasurement) || isempty(nearestNeighbors)
        defaultNearestNeighbors = floor(sqrt(size(trainX, 2))./2);
        nearestNeighbors = defaultNearestNeighbors;
        defaultDistanceMeasurement = 2;
        distanceMeasurement = defaultDistanceMeasurement;
    end;
    
    if nargin ==4 || isempty(distanceMeasurement)
        defaultDistanceMeasurement = 2;
        distanceMeasurement = defaultDistanceMeasurement;
        if nearestNeighbors > floor(sqrt(size(trainX, 2))./2)
            nearestNeighbors = floor(sqrt(size(trainX, 2))./2);
        end;
    end;
    
    outputArgs.nearestNeighbors = nearestNeighbors;
    outputArgs.distanceMeasurement = distanceMeasurement;
    estimatedClass = zeros(size(testData, 2),1);
    
    for ii =1:1:size(testData, 2)
        distanceVector = sum((abs(repmat(testData(:,ii),1,size(trainX, 2)) - trainX)).^(distanceMeasurement));
        [~, index] = sort(distanceVector, 2, 'ascend');
        if (mean(trainY(index(1:nearestNeighbors)))>= 0.5)
            estimatedClass(ii, 1) = 1;
        else
            estimatedClass(ii, 1) = 0;
        end;        
    end;
    
    outputArgs.estimatedClass = estimatedClass;
end