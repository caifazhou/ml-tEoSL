function [zScore] = zScoreCalculator(LSCoef, trainX, RSS)
% function [] = zScoreCalculator(coef, trainX, RSS)
% 
% compute the z score
% inputs
%   LSCoef: coefficients of LS
%   trainX: training examples for estimating the LS coefficients
%   RSS: residual sum of squares

% outputs
%   zScore:
[numberFeatures, numberTrain] = size(trainX);

varianceOfPrediction = RSS./(numberTrain - (numberFeatures + 1));
covarianceOfLSCoef = pinv(trainX*trainX').*varianceOfPrediction;
varianceOfLSCoef = diag(covarianceOfLSCoef);
zScore = LSCoef./(sqrt(varianceOfLSCoef).*(sqrt(diag(pinv(trainX*trainX')))));