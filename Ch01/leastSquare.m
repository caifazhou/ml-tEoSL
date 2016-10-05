function [ outputArgs ] = leastSquare( trainX, trainY )
% LEASTSQUARE linear least square estimation
% [ outputArgs ] = leastSquare( trainX, trainY )
% each column is a data point;
% input variables:
%   trainX: 
%   trainY:


% output variables (as a struct):
%   LS parameters: step
%   linearParameters: beta

% Author: CZ
% Version: 1.0
% Date              Status
% 26.09.2016        Draft

    if nargin ~= 2
       error('The number of inputs does not match!');
    else
        if (size(trainX, 2) ~= size(trainY, 2))
            error('The number of examples or the dimension of inputs does not match!');
        end;
    end;
    
    outputArgs.beta = pinv(trainX*(trainX'))*(trainX*(trainY'));
end

