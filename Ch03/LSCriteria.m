function [MSELoss] = LSCriteria(XTrain,YTrain,XTest,YTest)

addpath('..\Ch01\');
LSPara = leastSquare(XTrain,YTrain);
LSCoef = LSPara.beta;

YPredict = LSDecision(XTest, LSCoef, 0.5);
MSELoss = sum((YPredict.estimatedClass' - YTest).^2)./(size(XTest,2));
