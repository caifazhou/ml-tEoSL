function [sequentialSelectedFeature] = forwardSubsetSelection(fun, cv, features, targets, numberSelectedFeatures)
% function [] = forwardSubsetSelection(fun, cv, features, targets, numberSelectedFeatures)
% forward subset selection
% 
% inputs
% fun: a function handle takes cross validation datasets as input (train &
% test) an gives mean cross validation error as ouput;
% cv: cross validation partition object;
% features: features for cross validation (both train and test)
% targets: labels for cross validation 
% numberSelectedFeatures:
% output
% sequentialSelectedFeature: a struct includes
%     numberFeatures: for 0 to numberSelectedFeatures
%     selectedFeatureID
%     CVLoss: error of validation

if nargin < 4
    error('The number of inputs does not match!');
end;

if nargin == 4 || (numberSelectedFeatures > size(features, 1))
    numberSelectedFeatures = size(features, 1);
end;

numberFeatures = zeros(numberSelectedFeatures+1, 1);
selectedFeatureID = cell(numberSelectedFeatures+1, 1);
CVLoss = zeros(numberSelectedFeatures+1, 1);

% no feature is selected
numberFeatures(1,1) = 0;
selectedFeatureID{1,1} = 0;
% CVLoss(1,1) = inf;
% compute CVLoss without feature selected
temp = 0;
for ii = 1:1:cv.NumTestSets
    temp = temp + sum(targets(cv.test(ii)))./(cv.TestSize(ii));
end;
CVLoss(1,1) = temp./(cv.NumTestSets);

% forward selection
selectedList = [];
leftList = [1:1:size(features, 1)]';

for ii = 1:1:numberSelectedFeatures
%     greedy algorithm
    tempError = zeros(size(leftList,1), 1);
    for jj = 1:1:size(leftList,1)
        tempSelected = cat(1, selectedList, leftList(jj));
        tempError(jj, 1) = LSCrossValidation(fun, cv, features(tempSelected,:), targets);        
    end;
    [minLoss, minID] = min(tempError);
    selectedList = cat(1, selectedList, leftList(minID));
    numberFeatures(ii+1,1) = ii;
    selectedFeatureID{ii+1,1} = selectedList;
    leftList(minID) = [];
    CVLoss(ii+1, 1) = minLoss;
end;

% output
sequentialSelectedFeature.numberFeatures = numberFeatures;
sequentialSelectedFeature.selectedFeatureID = selectedFeatureID;
sequentialSelectedFeature.CVLoss = CVLoss;