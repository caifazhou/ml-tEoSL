function [CVError] = LSCrossValidation(fun, cv, X, y)
tempError = zeros(cv.NumTestSets,1);
for ii = 1:1:cv.NumTestSets
    trainID = cv.training(ii);
    testID = cv.test(ii);
    XTrain = X(:, trainID);
    XTrain = cat(1, ones(1, size(XTrain, 2)), XTrain);
    yTrain = y(:, trainID);
    XTest = X(:, testID);
    XTest = cat(1, ones(1, size(XTest, 2)), XTest);
    yTest = y(:, testID);
    tempError(ii, 1) = fun(XTrain, yTrain, XTest, yTest);
end;

CVError = sum(tempError)./(cv.NumTestSets);