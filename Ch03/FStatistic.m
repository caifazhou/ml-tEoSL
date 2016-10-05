function [FValue] = FStatistic(RSS0, RSS1, trainX0, trainX1)

if nargin ~= 4
    error('The number of inputs does not match');
else
    if size(trainX0,2) ~= size(trainX1,2)
        error('The number training examles of two sets does not equal!!');
    end;
    if (size(trainX0,2) <= (size(trainX0,2) +1))
        error('The number of examples is less than the number of features!');
    end;
end;


FValue = ((RSS0 - RSS1)./(size(trainX1,1) - size(trainX0,1)))./...
    (RSS1./(size(trainX0,2) - size(trainX1,1) - 1));