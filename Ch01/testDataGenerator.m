function [outputArgs] = testDataGenerator(trainX, step)

% To generate the test dataset according to the training data
% [outputArgs] = testDataGenerator(trainX, step)
% each column is a data point;
% input variables:
%   trainX: 
%   step:


% output variables (as a struct):
%   linearParameters: beta

% Author: CZ
% Version: 1.0
% Date              Status
% 26.09.2016        Draft

    if nargin < 1 || nargin > 2
        error('The number of inputs does not match');
    end;
    if nargin == 1
        defaultStep = 0.05;
        step = defaultStep;
    end;
    
    outputArgs.para.step = step;
    numberOfTestData = 1;
    
    for ii = 1:1:size(trainX, 1)
        numberOfTestData = numberOfTestData * (((ceil(max(trainX(ii,:))) - floor(min(trainX(ii,:))))./step) + 1);
    end;
    
    testDataset = zeros(size(trainX, 1), numberOfTestData);
    temp = cell(1,size(trainX ,1));
    
    for ii =1:1:size(trainX ,1)
        temp{1,ii} = floor(min(trainX(ii,:))):step:ceil(max(trainX(ii,:)));
    end;
    
    [X,Y] = meshgrid(temp{1,1},temp{1,2});
    testDataset(1,:) = reshape(X, 1, []);
    testDataset(2,:) = reshape(Y, 1, []);
    
    outputArgs.testDataset = testDataset;
    outputArgs.X = X;
    outputArgs.Y = Y;
    
end