% function [] = statisticalAnalysisProstate()

load('prostate.mat');
load('prostateTrain.mat');
target = strToBool(table2cell(prostateTrain))';

prostateValue = prostate(:,2:end)'; %a column is an example
% % correlation coefficients
prostateCorrCoef = corrcoef(prostate(:,2:end)');

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


