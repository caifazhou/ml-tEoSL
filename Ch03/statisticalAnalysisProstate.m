% function [] = statisticalAnalysisProstate()

load('prostate.mat');
load('prostateTrain.mat');
target = strToBool(table2cell(prostateTrain))';

prostateValue = prostate(:,2:end)'; %a column is an example
%%
% % correlation coefficients
prostateCorrCoef = corrcoef(prostate(:,2:end)');

%%
% % random split prostate into train and 
[numberFeatures, totalExamples] = size(prostateValue);
numberTrain = 67;
numberTest = totalExamples - numberTrain;

trainExamples = zeros(numberFeatures, numberTrain);
trainTarget = zeros(1, numberTrain);
testExamples = zeros(numberFeatures, numberTest);
testTarget = zeros(1, numberTest);

index = randperm(totalExamples);
for ii = 1:1:numberTrain
    trainExamples(:, ii) = prostateValue(:, index(1, ii));
    trainTarget(1, ii) = target(1, index(1, ii));
end;

for ii = 1:1:numberTest
    testExamples(:,ii) = prostateValue(:, index(1, ii + numberTrain));
    testTarget(:,ii) = target(:, index(1, ii + numberTrain));
end;

%%
% % compute z score
stdValue = std(prostate, 0, 1);

% using LS 
addpath('..\Ch01\');

trainX = cat(1, ones(1, size(trainExamples, 2)), trainExamples);
trainY = trainTarget;
LSParaProstate = leastSquare(trainX, trainY);
LSCoef = LSParaProstate.beta;
predictTarget = LSDecision(cat(1, ones(1, size(testExamples, 2)), testExamples), LSCoef, 0.5);
residualSumOfSqaures = sum((testTarget - predictTarget.estimatedClass').^2);
varianceOfPrediction = residualSumOfSqaures./(numberTrain - (numberFeatures + 1));
covarianceOfLSCoef = pinv(trainX*trainX').*varianceOfPrediction;
varianceOfLSCoef = diag(covarianceOfLSCoef);
zScore = LSCoef./(sqrt(varianceOfLSCoef).*(sqrt(diag(pinv(trainX*trainX')))));

%%
% subset selection & cross validation
totalExamples = prostate(:,2:end)';
% totalExamples = cat(1, ones(1, size(totalExamples, 2)), totalExamples);
totalTarget = target;

crossValidation = cvpartition(totalTarget,'KFold',10);
% opts = statset('display');
LSFun = @(XT,yT,Xt,yt)...
    (LSCriteria(XT,yT,Xt,yt));
CvError = LSCrossValidation(LSFun, crossValidation, totalExamples, totalTarget);

%%
% forward feature selection
forwardSelectionOutput = forwardSubsetSelection(LSFun, crossValidation, totalExamples, totalTarget);
% find the best subset
[~,bestSubsetID] = min(forwardSelectionOutput.CVLoss);
subsetFeatures = totalExamples(forwardSelectionOutput.selectedFeatureID{bestSubsetID},:);

% using LS to test the subset selection
addpath('..\Ch01\');

selectedFeatureID = forwardSelectionOutput.selectedFeatureID{bestSubsetID};
trainX = cat(1, ones(1, size(trainExamples, 2)), trainExamples(selectedFeatureID,:));
trainY = trainTarget;
subsetLSParaProstate = leastSquare(trainX, trainY);
subsetLSCoef = subsetLSParaProstate.beta;
subsetpredictTarget = LSDecision(cat(1, ones(1, size(testExamples, 2)), testExamples(selectedFeatureID,:)), subsetLSCoef, 0.5);
residualSumOfSqaures = sum((testTarget - subsetpredictTarget.estimatedClass').^2);
subsetvarianceOfPrediction = residualSumOfSqaures./(numberTrain - (size(selectedFeatureID,1) + 1));
subsetcovarianceOfLSCoef = pinv(trainX*trainX').*subsetvarianceOfPrediction;
subsetvarianceOfLSCoef = diag(subsetcovarianceOfLSCoef);
subsetZScore = subsetLSCoef./(sqrt(subsetvarianceOfLSCoef).*(sqrt(diag(pinv(trainX*trainX')))));
